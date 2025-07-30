# This helper functions helpe keep track of training metrics and sample diversity

import torch, matplotlib.pyplot as plt
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from monai.metrics import DiceMetric
from typing import Dict, Union
import numpy as np 

eps, _active_eps = 1e-7, 1e-3   # constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TrainingMetrics:
    """Tracks losses, dice/accuracy, KL, and true active latent dims across an epoch."""
    
    def __init__(self, act_thresh: float = 1e-3):
        self.act_thresh = act_thresh
        self.reset_epoch()
    
    def reset_epoch(self):
        self.total_loss = 0.0
        self.total_dice = 0.0
        self.total_acc  = 0.0
        self.total_kl   = 0.0
        self.batch_count = 0
        self._all_mu_q = []   # stores mu_q for every batch

    def update_batch(self, logits, targets, kl_loss, total_loss):
        # usual metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dice = self._compute_dice(preds, targets)
        acc  = preds.eq(targets).float().mean()

        self.total_loss += total_loss.item()
        self.total_dice += dice.item()
        self.total_acc  += acc.item()
        self.total_kl   += kl_loss.item()
        self.batch_count += 1

    def update_latents(self, mu_q: torch.Tensor):
        # Collect every batch's posterior means
        # mu_q: [B, latent_dim]
        self._all_mu_q.append(mu_q.detach().cpu())

    def finalize(self, batch_size):
        """Return averaged metrics for the epoch, including true active dims."""
        # basic averages
        stats = {
            'loss': self.total_loss  / max(self.batch_count, 1),
            'dice': self.total_dice  / max(self.batch_count, 1),
            'acc':  self.total_acc   / max(self.batch_count, 1),
            'kl':   self.total_kl    / max(self.batch_count, 1),
        }

        # compute active dims over ALL collected mu_q
        if self._all_mu_q:
            all_mu = torch.cat(self._all_mu_q, dim=0)       # [N_samples, latent_dim]
            var   = all_mu.var(dim=0)                       # [latent_dim]
            act_dims = int((var > self.act_thresh).sum().item())
        else:
            act_dims = 0
        
        stats['active_dims'] = act_dims
        return stats

    def _compute_dice(self, preds, targets, smooth=1e-6):
        p = preds.view(preds.size(0), -1)
        t = targets.view(targets.size(0), -1)
        inter = (p * t).sum(dim=1)
        return ((2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()

def _sample_diversity(model, x_sample, n_samples=8):
    """Compute prediction diversity for probabilistic models."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_sample)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            pred = (torch.sigmoid(logits) > 0.5).float()
            samples.append(pred.cpu())
    
    # Compute diversity as std of predictions
    samples = torch.stack(samples, dim=0)  # [n_samples, B, C, D, H, W]
    diversity = samples.float().std(dim=0).mean().item()
    return diversity