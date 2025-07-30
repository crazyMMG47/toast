#### This module contains the metric functions used in training and evaluating the model.
# Author: Hailin Liang 
# GED reference: see probabilistic unet paper 

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
import numpy as np

# Example for binary:
dice_loss = DiceLoss(sigmoid=True)  # expects raw logits -> applies sigmoid internally
bce_loss = torch.nn.BCEWithLogitsLoss()

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Tuple, Any
from monai.metrics import DiceMetric


def compute_slice_dice(pred_slice, target_slice, smooth=1e-8):
    """
    Compute Dice coefficient for a single 2D slice.
    
    Args:
        pred_slice: Predicted mask slice (2D tensor)
        target_slice: Ground truth mask slice (2D tensor)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Dice coefficient as a tensor (not float)
    """
    # Flatten the slices
    pred_flat = pred_slice.view(-1)
    target_flat = target_slice.view(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # Compute Dice coefficient and ensure it's a tensor
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Make sure we return a tensor, not a scalar
    return dice.detach()  # detach to avoid gradient computation issues


def dice_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dice loss implementation."""
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    
    # Flatten
    probs_flat = probs.view(probs.size(0), -1)
    mask_flat = mask.view(mask.size(0), -1)
    
    intersection = (probs_flat * mask_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (probs_flat.sum(dim=1) + mask_flat.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def bce_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss."""
    return nn.functional.binary_cross_entropy_with_logits(logits, mask)


def recon_loss_fn(logits: torch.Tensor, mask: torch.Tensor, bce_weight: float = 0.3) -> torch.Tensor:
    """Combined Dice and BCE reconstruction loss."""
    loss_dice = dice_loss(logits, mask)
    loss_bce = bce_loss(logits, mask)
    return loss_dice + bce_weight * loss_bce


def kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                  mu_p: torch.Tensor, logvar_p: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
    """Compute KL(q||p) for two diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    kl_elem = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
    kl = kl_elem.sum(dim=1)  # [B]
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        return kl
    
class SliceKLWeighting:
    """Reweights only the bottom `worst_slice_percentage` of slices by Dice."""

    def __init__(
        self,
        worst_slice_percentage: float = 0.1,
        w_hard: float = 2.0,
        w_other: float = 1.0,
    ):
        self.worst_slice_percentage = worst_slice_percentage
        self.w_hard = w_hard
        self.w_other = w_other

    def __call__(
        self,
        model_output: Tuple,
        logits: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Unpack posterior & prior
        _, (mu_p, logvar_p), (mu_q, logvar_q) = model_output
        B, _, D = logits.shape[:3]

        # B×D slice‐wise Dice
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        total_weighted_kl = 0.0
        batch_weights = []

        n_worst = max(1, int(D * self.worst_slice_percentage))

        for b in range(B):
            # 1) compute each slice's Dice
            dice_scores = []
            for d in range(D):
                dice_scores.append(compute_slice_dice(preds[b, :, d], y[b, :, d]))
            dice_scores = torch.tensor(dice_scores, device=logits.device)

            # 2) find the indices of the n_worst lowest‐Dice slices
            worst_idxs = torch.topk(dice_scores, n_worst, largest=False).indices

            # 3) compute a single scalar weight for this sample:
            #    average of w_hard on those worst slices, w_other on the rest
            subject_weight = (
                self.w_hard * n_worst
                + self.w_other * (D - n_worst)
            ) / D

            # 4) compute (scalar) KL for this sample
            #    (uses default reduction='batch' → one value per b)
            subject_kl = kl_divergence(
                mu_q[b : b + 1],
                logvar_q[b : b + 1],
                mu_p[b : b + 1],
                logvar_p[b : b + 1],
            )

            total_weighted_kl += subject_weight * subject_kl
            batch_weights.append(subject_weight)

        # 5) final loss & diagnostics
        final_kl = total_weighted_kl / B
        diagnostics = {
            "w_bar": float(np.mean(batch_weights)),         # avg weight
            "kl_vol": final_kl.item(),                      # weighted KL
            "worst_pct": self.worst_slice_percentage * 100, # always 10%
        }
        return final_kl, diagnostics
    
class SliceKLWeighting_Version2:
    """Handler for slice-level KL divergence weighting based on reconstruction difficulty."""
    
    def __init__(self, worst_slice_percentage: float = 0.1, w_hard: float = 2.0, 
                 w_other: float = 1.0, dice_threshold: float = 0.3):
        self.worst_slice_percentage = worst_slice_percentage
        self.w_hard = w_hard
        self.w_other = w_other
        self.dice_threshold = dice_threshold
    
    def __call__(self, model_output: Tuple, logits: torch.Tensor, 
                 y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute weighted KL loss based on slice-level difficulty analysis.
        
        Args:
            model_output: (logits, (mu_p, logvar_p), (mu_q, logvar_q))
            logits: [B, C, D, H, W]
            y: [B, C, D, H, W]
            
        Returns:
            weighted_kl_loss: Final weighted KL divergence
            diagnostics: Dictionary with metrics and diagnostics
        """
        _, (mu_p, logvar_p), (mu_q, logvar_q) = model_output
        batch_size, _, depth = logits.shape[:3]
        
        # Convert to predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Initialize tracking
        total_weighted_kl = 0.0
        all_weights = []
        all_hard_ratios = []
        total_hard_slices = 0
        total_slices = 0
        
        for b in range(batch_size):
            # Compute per-slice Dice scores for this subject
            slice_dice_scores = []
            for d in range(depth):
                dice_score = compute_slice_dice(preds[b, :, d], y[b, :, d])
                slice_dice_scores.append(dice_score)
            
            # slice_dice_scores = torch.stack(slice_dice_scores)
            slice_dice_scores = torch.tensor(slice_dice_scores, device=logits.device)
            
            # Identify worst slices (lowest Dice scores)
            n_worst_slices = max(1, int(depth * self.worst_slice_percentage))
            n_worst_slices = min(n_worst_slices, depth)
            
            worst_indices = torch.topk(slice_dice_scores, n_worst_slices, largest=False).indices
            worst_dice_scores = slice_dice_scores[worst_indices]
            
            # Determine hard slices (below threshold)
            hard_slice_mask = worst_dice_scores < self.dice_threshold
            n_hard_slices = hard_slice_mask.sum().item()
            hard_ratio = n_hard_slices / n_worst_slices if n_worst_slices > 0 else 0.0
            
            # Compute subject-level weight
            if n_hard_slices > 0:
                # Interpolate between w_other and w_hard based on hard slice ratio
                subject_weight = self.w_other + (self.w_hard - self.w_other) * hard_ratio
            else:
                subject_weight = self.w_other
            
            # Apply weight to this subject's KL divergence
            subject_kl = kl_divergence(
                mu_q[b:b+1], logvar_q[b:b+1], 
                mu_p[b:b+1], logvar_p[b:b+1], 
                reduction="none"
            )
            
            weighted_subject_kl = subject_weight * subject_kl
            total_weighted_kl += weighted_subject_kl
            
            # Track statistics
            all_weights.append(subject_weight)
            all_hard_ratios.append(hard_ratio)
            total_hard_slices += n_hard_slices
            total_slices += n_worst_slices
        
        # Final weighted KL loss
        final_kl_loss = total_weighted_kl / batch_size
        
        # Compute diagnostics
        diagnostics = {
            'w_bar': np.mean(all_weights),  # Average weight across batch
            'kl_vol': final_kl_loss.item(),  # Volume-level KL after weighting
            'hard_pct': (total_hard_slices / max(total_slices, 1)) * 100,  # % hard slices
            'subject_weights': all_weights,
            'hard_ratios': all_hard_ratios,
            'mean_hard_ratio': np.mean(all_hard_ratios),
        }
        
        return final_kl_loss, diagnostics

    
# Calculate the slice-wise dice score per subject. 
def compute_slice_dice(pred_slice, target_slice, smooth=1e-6):
    """
    Compute Dice score for a single 2D slice.
    
    Args:
        pred_slice: predicted segmentation [H, W] or [C, H, W]
        target_slice: target segmentation [H, W] or [C, H, W]
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        dice_score: Dice coefficient as float
    """
    if pred_slice.dim() > 2:
        pred_slice = pred_slice.flatten()
        target_slice = target_slice.flatten()
    else:
        pred_slice = pred_slice.flatten()
        target_slice = target_slice.flatten()
    
    intersection = (pred_slice * target_slice).sum()
    dice = (2.0 * intersection + smooth) / (pred_slice.sum() + target_slice.sum() + smooth)
    return dice.item()


# find the slices with the 10 lowest dice score
def has_hard_slice(logits, y, method="lowest_dice", **kwargs):
    """
    Determine which volumes in the batch contain "hard" slices based on Dice scores.
    
    Args:
        logits: model predictions [B, C, D, H, W] or [B, C, H, W]  
        y: target tensor [B, C, D, H, W] or [B, C, H, W]
        method: method to identify hard slices
            - "lowest_dice": select volumes containing the worst Dice slices
        **kwargs: additional parameters
            - n_worst_slices: number of worst slices to consider per volume (default: 10)
            - dice_threshold: threshold below which slices are considered hard (default: 0.3)
    
    Returns:
        mask_hard: boolean tensor [B] indicating which volumes have hard slices
    """
    batch_size = logits.size(0)
    mask_hard = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
    
    if method == "lowest_dice":
        n_worst_slices = kwargs.get("n_worst_slices", 10)
        dice_threshold = kwargs.get("dice_threshold", 0.3)
        
        # Convert logits to predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        if logits.dim() == 5:  # 3D volumes [B, C, D, H, W]
            for b in range(batch_size):
                slice_dice_scores = []
                depth = logits.size(2)
                
                # Compute Dice for each slice
                for d in range(depth):
                    pred_slice = preds[b, :, d]  # [C, H, W]
                    target_slice = y[b, :, d]    # [C, H, W]
                    dice_score = compute_slice_dice(pred_slice, target_slice)
                    slice_dice_scores.append(dice_score)
                
                # Find the n worst slices
                slice_dice_scores = torch.tensor(slice_dice_scores)
                n_consider = min(n_worst_slices, len(slice_dice_scores))
                worst_dice_scores = torch.topk(slice_dice_scores, n_consider, largest=False).values
                
                # Mark as hard if any of the worst slices is below threshold
                mask_hard[b] = worst_dice_scores.min() < dice_threshold
                
        else:  # 2D slices [B, C, H, W] - each sample is already a slice
            for b in range(batch_size):
                pred_slice = preds[b]   # [C, H, W]
                target_slice = y[b]     # [C, H, W]
                dice_score = compute_slice_dice(pred_slice, target_slice)
                mask_hard[b] = dice_score < dice_threshold
    
    return mask_hard