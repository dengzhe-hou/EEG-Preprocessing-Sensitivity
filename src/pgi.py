"""
Pipeline Generator Invariance (PGI) training.

Core learning principle: suppress prediction drift on atomic intervention
edges of the preprocessing semigroup. Telescoping bound guarantees that
local edge invariance implies global composition invariance.
"""

import torch
import torch.nn.functional as F

from .intervention_graph import (
    build_hasse_edges, get_edge_tensors, compute_edge_weights, K
)


class PGITrainer:
    """Wraps any EEG decoder with PGI training objective.

    The PGI loss penalizes logit differences between pipeline views
    connected by atomic intervention edges in the Hasse diagram.
    """

    def __init__(self, model, optimizer, n_classes,
                 pgi_lambda=1.0, edge_weight_mode="perturbation",
                 warmup_epochs=5, device="cuda",
                 pipeline_indices=None,
                 normalize_pgi=False, adaptive_lambda=False,
                 cfr_target=0.15):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.pgi_lambda = pgi_lambda
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.normalize_pgi = normalize_pgi
        self.adaptive_lambda = adaptive_lambda
        self.cfr_target = cfr_target
        self.running_cfr = cfr_target  # initialize to target (not 0!)

        # Build intervention graph, optionally restricted to a subset
        all_edges = build_hasse_edges()

        if pipeline_indices is not None:
            # Map original pipeline IDs to contiguous view indices
            pi_set = set(pipeline_indices)
            pi_to_view = {pi: i for i, pi in enumerate(sorted(pi_set))}
            # Keep only edges where both endpoints are in the subset
            edges = [e for e in all_edges if e.src in pi_set and e.dst in pi_set]
            src = torch.tensor([pi_to_view[e.src] for e in edges], dtype=torch.long)
            dst = torch.tensor([pi_to_view[e.dst] for e in edges], dtype=torch.long)
            iids = torch.tensor([e.intervention for e in edges], dtype=torch.long)
        else:
            edges = all_edges
            src, dst, iids = get_edge_tensors(edges)

        self.src_idx = src.to(device)
        self.dst_idx = dst.to(device)
        self.intervention_ids = iids.to(device)

        # Edge weights (per intervention type)
        intervention_weights = compute_edge_weights(mode=edge_weight_mode)
        self.edge_weights = intervention_weights[iids].to(device)

    def get_lambda(self, epoch):
        """Linear warmup of PGI lambda."""
        if self.warmup_epochs <= 0:
            return self.pgi_lambda
        return self.pgi_lambda * min(1.0, epoch / self.warmup_epochs)

    def train_step(self, batch_views, labels, epoch=0, n_edge_sample=32,
                   n_sup_views=8):
        """Single training step with PGI loss.

        With 128 pipelines, we subsample for memory efficiency:
        - Supervised loss: on n_sup_views random pipeline views
        - PGI loss: on n_edge_sample random edges from the Hasse graph

        Args:
            batch_views: (B, V, C, T) tensor — V pipeline views per trial
            labels: (B,) tensor of class labels
            epoch: current epoch for lambda warmup
            n_edge_sample: number of Hasse edges to sample per step (0=all)
            n_sup_views: number of views for supervised loss (0=all)
        """
        self.model.train()
        B, V, C, T = batch_views.shape
        E = len(self.src_idx)

        # Sample edges for PGI loss
        if n_edge_sample > 0 and n_edge_sample < E:
            edge_perm = torch.randperm(E, device=self.device)[:n_edge_sample]
            s_src = self.src_idx[edge_perm]
            s_dst = self.dst_idx[edge_perm]
            s_weights = self.edge_weights[edge_perm]
        else:
            s_src, s_dst, s_weights = self.src_idx, self.dst_idx, self.edge_weights

        # Determine which pipeline views we need to forward-pass
        needed_views = set()
        needed_views.update(s_src.cpu().tolist())
        needed_views.update(s_dst.cpu().tolist())

        # Add random views for supervised loss
        if n_sup_views > 0 and n_sup_views < V:
            sup_views = torch.randperm(V)[:n_sup_views].tolist()
        else:
            sup_views = list(range(V))
        needed_views.update(sup_views)

        needed_views = sorted(needed_views)
        view_to_local = {v: i for i, v in enumerate(needed_views)}
        n_forward = len(needed_views)

        # Forward pass only needed views
        selected = batch_views[:, needed_views]  # (B, n_forward, C, T)
        logits = self.model(selected.reshape(B * n_forward, C, T)).reshape(
            B, n_forward, self.n_classes)

        # Supervised loss on sup_views
        sup_local = [view_to_local[v] for v in sup_views]
        sup_logits = logits[:, sup_local]  # (B, n_sup, n_classes)
        n_s = len(sup_local)
        sup_loss = F.cross_entropy(
            sup_logits.reshape(B * n_s, self.n_classes),
            labels.repeat_interleave(n_s)
        )

        # PGI loss on sampled edges
        local_src = torch.tensor([view_to_local[v] for v in s_src.cpu().tolist()],
                                  device=self.device)
        local_dst = torch.tensor([view_to_local[v] for v in s_dst.cpu().tolist()],
                                  device=self.device)
        z = logits - logits.mean(dim=-1, keepdim=True)
        edge_diffs = (z[:, local_src] - z[:, local_dst]).pow(2).sum(-1)
        pgi_loss_raw = (edge_diffs * s_weights.unsqueeze(0)).mean()

        # Fix 1: Normalize by logit variance (scale-invariant across datasets)
        # IMPORTANT: detach denominator to prevent gradient from exploding logits
        if self.normalize_pgi:
            logit_var = z.detach().pow(2).mean() + 1e-8
            pgi_loss = pgi_loss_raw / logit_var
        else:
            pgi_loss = pgi_loss_raw

        # Fix 2: Adaptive lambda based on running CFR
        lam = self.get_lambda(epoch)
        if self.adaptive_lambda:
            cfr_ratio = max(0.01, min(5.0, self.running_cfr / (self.cfr_target + 1e-8)))
            lam = lam * cfr_ratio

        total_loss = sup_loss + lam * pgi_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = logits[:, view_to_local.get(0, 0)].argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "pgi_loss": pgi_loss.item(),
            "lambda": lam,
            "accuracy": acc,
            "n_views_forwarded": n_forward,
        }

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        """Evaluation step: compute accuracy and flip rates.

        Args:
            batch_views: (B, V, C, T)
            labels: (B,)

        Returns:
            dict with metrics
        """
        self.model.eval()
        B, V, C, T = batch_views.shape

        flat_views = batch_views.reshape(B * V, C, T)
        logits = self.model(flat_views).reshape(B, V, self.n_classes)
        preds = logits.argmax(dim=-1)  # (B, V)

        # Per-pipeline accuracy
        accs = []
        for v in range(V):
            acc = (preds[:, v] == labels).float().mean().item()
            accs.append(acc)

        # Counterfactual flip rate
        ref_preds = preds[:, 0:1].expand_as(preds)  # compare all to pipeline 0
        flips = (preds != ref_preds).float()
        cfr = flips.mean().item()
        max_cfr = flips.mean(dim=0).max().item()

        # Update running CFR for adaptive lambda
        if self.adaptive_lambda:
            self.running_cfr = 0.9 * self.running_cfr + 0.1 * cfr

        # PGI loss (for monitoring) — only if V matches edge index range
        max_edge_idx = max(self.src_idx.max().item(), self.dst_idx.max().item())
        if V > max_edge_idx:
            z = logits - logits.mean(dim=-1, keepdim=True)
            edge_diffs = (z[:, self.src_idx] - z[:, self.dst_idx]).pow(2).sum(-1)
            pgi_loss = (edge_diffs * self.edge_weights.unsqueeze(0)).mean().item()
        else:
            pgi_loss = float('nan')

        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "best_accuracy": max(accs),
            "cfr": cfr,
            "max_cfr": max_cfr,
            "pgi_loss": pgi_loss,
        }


class ERMTrainer:
    """Standard ERM baseline (single-pipeline or mixed-pipeline)."""

    def __init__(self, model, optimizer, n_classes, mixed=False, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.mixed = mixed
        self.device = device

    def train_step(self, batch_views, labels, epoch=0):
        self.model.train()
        B, V, C, T = batch_views.shape

        if self.mixed:
            # ERM-mixed: train on random pipeline per sample
            idx = torch.randint(V, (B,), device=batch_views.device)
            x = batch_views[torch.arange(B), idx]
        else:
            # ERM-single: train on pipeline 0 only
            x = batch_views[:, 0]

        logits = self.model(x)
        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(-1) == labels).float().mean().item()

        return {"loss": loss.item(), "accuracy": acc}

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        self.model.eval()
        B, V, C, T = batch_views.shape

        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)
        preds = logits.argmax(-1)

        accs = [(preds[:, v] == labels).float().mean().item() for v in range(V)]
        ref_preds = preds[:, 0:1].expand_as(preds)
        flips = (preds != ref_preds).float()

        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "cfr": flips.mean().item(),
            "max_cfr": flips.mean(dim=0).max().item(),
        }


class PairedConsistencyTrainer:
    """Paired consistency regularization across pipeline views."""

    def __init__(self, model, optimizer, n_classes,
                 consistency_lambda=1.0, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.consistency_lambda = consistency_lambda
        self.device = device

    def train_step(self, batch_views, labels, epoch=0):
        self.model.train()
        B, V, C, T = batch_views.shape

        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)

        # Supervised on all views
        sup_loss = F.cross_entropy(
            logits.reshape(B * V, self.n_classes),
            labels.repeat_interleave(V)
        )

        # Pairwise consistency: all views should agree
        mean_logits = logits.mean(dim=1, keepdim=True)
        consistency_loss = ((logits - mean_logits) ** 2).mean()

        loss = sup_loss + self.consistency_lambda * consistency_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = (logits[:, 0].argmax(-1) == labels).float().mean().item()

        return {
            "loss": loss.item(),
            "sup_loss": sup_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "accuracy": acc,
        }

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        self.model.eval()
        B, V, C, T = batch_views.shape
        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)
        preds = logits.argmax(-1)
        accs = [(preds[:, v] == labels).float().mean().item() for v in range(V)]
        ref_preds = preds[:, 0:1].expand_as(preds)
        flips = (preds != ref_preds).float()
        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "cfr": flips.mean().item(),
        }


class GroupDROTrainer:
    """GroupDRO: minimize worst-group (pipeline) loss."""

    def __init__(self, model, optimizer, n_classes, eta=0.01,
                 n_views_sample=8, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.eta = eta
        self.n_views_sample = n_views_sample
        self.device = device
        self.q = None  # initialized lazily

    def train_step(self, batch_views, labels, epoch=0):
        self.model.train()
        B, V, C, T = batch_views.shape

        # Subsample views for memory
        if self.n_views_sample < V:
            idx = torch.randperm(V)[:self.n_views_sample]
            views = batch_views[:, idx]
            n_v = self.n_views_sample
        else:
            views = batch_views
            n_v = V

        flat = views.reshape(B * n_v, C, T)
        logits = self.model(flat).reshape(B, n_v, self.n_classes)

        # Per-view losses
        per_view_loss = torch.zeros(n_v, device=self.device)
        for v in range(n_v):
            per_view_loss[v] = F.cross_entropy(logits[:, v], labels)

        # Update group weights via exponentiated gradient
        if self.q is None:
            self.q = torch.ones(n_v, device=self.device) / n_v
        self.q = self.q * torch.exp(self.eta * per_view_loss.detach())
        self.q = self.q / self.q.sum()

        # Weighted loss (emphasizes worst group)
        loss = (self.q * per_view_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = (logits[:, 0].argmax(-1) == labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc,
                "worst_group_loss": per_view_loss.max().item()}

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        self.model.eval()
        B, V, C, T = batch_views.shape
        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)
        preds = logits.argmax(-1)
        accs = [(preds[:, v] == labels).float().mean().item() for v in range(V)]
        ref_preds = preds[:, 0:1].expand_as(preds)
        flips = (preds != ref_preds).float()
        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "cfr": flips.mean().item(),
            "max_cfr": flips.mean(dim=0).max().item(),
        }


class IRMTrainer:
    """IRM: Invariant Risk Minimization with gradient penalty."""

    def __init__(self, model, optimizer, n_classes, irm_lambda=1.0,
                 n_views_sample=8, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.irm_lambda = irm_lambda
        self.n_views_sample = n_views_sample
        self.device = device

    def _irm_penalty(self, logits, labels):
        """Compute IRM gradient penalty: ||grad(w * CE)||^2 where w=1."""
        scale = torch.ones(1, device=self.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, labels)
        grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
        return grad.pow(2).sum()

    def train_step(self, batch_views, labels, epoch=0):
        self.model.train()
        B, V, C, T = batch_views.shape

        if self.n_views_sample < V:
            idx = torch.randperm(V)[:self.n_views_sample]
            views = batch_views[:, idx]
            n_v = self.n_views_sample
        else:
            views = batch_views
            n_v = V

        flat = views.reshape(B * n_v, C, T)
        logits = self.model(flat).reshape(B, n_v, self.n_classes)

        # ERM loss + IRM penalty per view
        erm_loss = 0
        irm_penalty = 0
        for v in range(n_v):
            erm_loss += F.cross_entropy(logits[:, v], labels)
            irm_penalty += self._irm_penalty(logits[:, v], labels)
        erm_loss /= n_v
        irm_penalty /= n_v

        loss = erm_loss + self.irm_lambda * irm_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = (logits[:, 0].argmax(-1) == labels).float().mean().item()
        return {"loss": loss.item(), "erm_loss": erm_loss.item(),
                "irm_penalty": irm_penalty.item(), "accuracy": acc}

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        self.model.eval()
        B, V, C, T = batch_views.shape
        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)
        preds = logits.argmax(-1)
        accs = [(preds[:, v] == labels).float().mean().item() for v in range(V)]
        ref_preds = preds[:, 0:1].expand_as(preds)
        flips = (preds != ref_preds).float()
        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "cfr": flips.mean().item(),
            "max_cfr": flips.mean(dim=0).max().item(),
        }


class CORALTrainer:
    """Deep CORAL: align feature covariance across pipeline views."""

    def __init__(self, model, optimizer, n_classes, coral_lambda=1.0,
                 n_views_sample=8, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.coral_lambda = coral_lambda
        self.n_views_sample = n_views_sample
        self.device = device

    def train_step(self, batch_views, labels, epoch=0):
        self.model.train()
        B, V, C, T = batch_views.shape

        if self.n_views_sample < V:
            idx = torch.randperm(V)[:self.n_views_sample]
            views = batch_views[:, idx]
            n_v = self.n_views_sample
        else:
            views = batch_views
            n_v = V

        flat = views.reshape(B * n_v, C, T)
        features = self.model.get_features(flat).reshape(B, n_v, -1)  # (B, V, D)
        logits = self.model.classifier(features.reshape(B * n_v, -1)).reshape(B, n_v, self.n_classes)

        # Supervised loss
        sup_loss = F.cross_entropy(
            logits.reshape(B * n_v, self.n_classes),
            labels.repeat_interleave(n_v)
        )

        # CORAL: align covariance matrices across views
        # Compute mean covariance and penalize deviations
        mean_feat = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        coral_loss = ((features - mean_feat).pow(2)).mean()

        loss = sup_loss + self.coral_lambda * coral_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = (logits[:, 0].argmax(-1) == labels).float().mean().item()
        return {"loss": loss.item(), "sup_loss": sup_loss.item(),
                "coral_loss": coral_loss.item(), "accuracy": acc}

    @torch.no_grad()
    def eval_step(self, batch_views, labels):
        self.model.eval()
        B, V, C, T = batch_views.shape
        flat = batch_views.reshape(B * V, C, T)
        logits = self.model(flat).reshape(B, V, self.n_classes)
        preds = logits.argmax(-1)
        accs = [(preds[:, v] == labels).float().mean().item() for v in range(V)]
        ref_preds = preds[:, 0:1].expand_as(preds)
        flips = (preds != ref_preds).float()
        return {
            "mean_accuracy": sum(accs) / len(accs),
            "worst_accuracy": min(accs),
            "cfr": flips.mean().item(),
            "max_cfr": flips.mean(dim=0).max().item(),
        }
