## This module will contains a set of evaluation metrics for the Prob U-Net model.
# Directory:
# (1) dice_hard -- computes the mean Dice over the hardest fraction of slices.
# (2) dice_score_batch -- computes the average Dice score over all slices.
# (3) GED -- computes the Generalized Energy Distance between a set of predictions and a reference mask.
# (4) S_NCC -- computes the Structural Normalized Cross-Correlation between a set of predictions and a reference mask.

import torch
from typing import Callable
import itertools


# This function computes the dice score over the hardest fraction of slices.
def dice_hard(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5, 
    eps: float = 1e-7,
    hard_frac: float = 0.1
) -> torch.Tensor:
    
    """
    Mean Dice over the hardest fraction of slices. It is now set to the lowest 10% of slices by Dice score.
    You should adjust this fraction based on their specific needs.

    Args:
    @logits: (N,1,H,W) or (N,1,D,H,W) – raw logits or probabilities
    @targets: same shape, binary ground‐truth masks {0,1}
    @threshold: binarisation cutoff if logits are probs 
    @eps: stability constant
    @hard_frac: fraction of slices to treat as “hard” (e.g. 0.1 for 10%) 

    Returns:
    @scalar Tensor: average Dice across the hardest slices
    """
    # logits are unconstrained real numbers and we binarize them using our defined "threshold"
    # use sigmoid to convert probabilities to [0, 1] range
    probs = logits.sigmoid()
    preds = (probs > threshold).float()
    # debug:
    # print(preds.shape, targets.shape)
    
    # Flattening each slice's spatial dimensions into a single vector let us computer its Dice score 
    # with a single dot-product / sum formula 
    if preds.ndim == 5: # 3D case
        B,_,D,H,W = preds.shape
        # flattern each slice to (B*D, H*W)
        preds_flat = preds.view(B*D, H*W)
        targets_flat = targets.view(B*D, H*W) # ground truth is also flattened per slice for comparison
        
    # also handles 2D case here (although we're not using it :< )
    elif preds.ndim == 4:
        B,_,H,W    = preds.shape
        preds_flat   = preds.view(B, H*W)
        targets_flat = targets.view(B, H*W)
        
    else:
        raise ValueError(f"Unsupported input dims {preds.shape}")

    # Computer dice score for each slice
    # element-wise AND operation -> intersection
    inter = (preds_flat * targets_flat).sum(dim=1)
    # sum of both predictions and targets PER SLICE -> union
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice_per = (2*inter + eps) / (union + eps) # dice score PER slice 

    # select the hardest slies 
    # Count num of slices we have 
    N = dice_per.numel()
    # choose the bottom k slices based on the hard_frac
    k = max(1, int(N * hard_frac))
    _, idx = torch.topk(-dice_per, k, sorted=False)

    # return the mean of the hardest slices
    return dice_per[idx].mean()


# Compute the dice score of all test subjects 
def dice_score_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-7,
    return_mean: bool = False
) -> torch.Tensor:
    """
    Calculates Dice score for a batch of predictions and targets.

    Args:
    @preds (N, …): Tensor of binary or probabilistic masks.
    @targets(N, …): Tensor of binary ground-truth masks.
    @epsilon: small constant to avoid div-by-zero.
    @return_mean: if True, returns a single scalar (mean over N subjects);
                   otherwise returns Tensor[N] of per-subject Dice.
    Returns:
      Tensor[N] or scalar
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {preds.shape} vs {targets.shape}")

    N = preds.shape[0]
    # flatten everything but batch dim
    flat_p = preds.contiguous().view(N, -1)
    flat_t = targets.contiguous().view(N, -1)

    inter = (flat_p * flat_t).sum(dim=1)  
    denom = flat_p.sum(dim=1) + flat_t.sum(dim=1)
    dice  = (2 * inter + epsilon) / (denom + epsilon)
    return dice.mean() if return_mean else dice


# D2GED
def deg(
    samples: torch.Tensor,
    reference: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    Generalized Energy Distance between a set of M predictions and a reference mask.
    
    GED = 2 * E[d(s, ref)] − E[d(s, s')]
    
    Args:
        samples: Tensor of shape (M, ...), M different segmentations
        reference: Tensor of shape (...), the ground-truth mask
        distance_fn: function taking two tensors (same shape) → scalar distance
    Returns:
        Scalar Tensor: the GED score
    """
    M = samples.shape[0]
    # average distance sample → reference
    d_sr = torch.stack([distance_fn(samples[i], reference) for i in range(M)]).mean()
    # average distance sample → sample
    d_ss = torch.stack([
        distance_fn(samples[i], samples[j])
        for i, j in itertools.product(range(M), repeat=2)
    ]).mean()
    return 2 * d_sr - d_ss


# S_NCC 
def s_ncc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Normalized cross-correlation between two spatial volumes/masks.
    
    Args:
        pred: Tensor of arbitrary shape (...).
        target: same shape as pred.
        eps: small constant to avoid division by zero.
    Returns:
        Scalar Tensor: NCC coefficient in [−1,1].
    """
    p = pred.view(-1).float()
    t = target.view(-1).float()
    mp, mt = p.mean(), t.mean()
    sp, st = p.std(unbiased=False) + eps, t.std(unbiased=False) + eps
    return ((p - mp) * (t - mt)).sum() / (sp * st * p.numel())


def deg_hard(
    samples: torch.Tensor,
    reference: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hard_frac: float = 0.1,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    GED over only the lowest‐Dice fraction of slices.
    
    Args:
      samples: (M, 1, D, H, W) or (M, 1, H, W)
      reference: (1, D, H, W) or (1, H, W)
      distance_fn: same as for deg()
      hard_frac: fraction of slices to treat as “hard”
    Returns:
      Scalar Tensor: GED computed on only those slices
    """
    M = samples.shape[0]
    # remove channel dim if present
    samp = samples.squeeze(1)       # → (M, D, H, W) or (M, H, W)
    ref  = reference.squeeze(0)     # → (D, H, W) or (H, W)

    # 1) compute per-slice avg dice across M samples
    D = samp.shape[1] if samp.ndim == 4 else 1
    dice_per_slice = []
    for d in range(D):
        # collect dice(sample_i[d], ref[d])
        dice_vals = []
        slice_ref = ref if D==1 else ref[d]
        for i in range(M):
            slice_pred = samp[i] if D==1 else samp[i,d]
            # your Dice score fn, but as distance_fn expects 1-dist:
            dice = 1 - distance_fn(slice_pred, slice_ref)
            dice_vals.append(dice)
        dice_per_slice.append(torch.stack(dice_vals).mean())
    dice_per_slice = torch.stack(dice_per_slice)

    # 2) pick worst k slices
    N = dice_per_slice.numel()
    k = max(1, int(N * hard_frac))
    worst_idxs = torch.topk(dice_per_slice, k, largest=False).indices

    # 3) build new samples & ref restricted to those slices
    if D == 1:
        hard_samps = samp.unsqueeze(1)          # back to (M,1,H,W)
        hard_ref   = ref.unsqueeze(0)           # (1,H,W)
    else:
        hard_samps = samp[:, worst_idxs, ...]   # (M, k, H, W)
        hard_ref   = ref[worst_idxs, ...]       # (k, H, W)

    # insert channel dim back for deg
    hard_samps = hard_samps.unsqueeze(1)       # (M,1,k,H,W) or (M,1,H,W)
    hard_ref   = hard_ref.unsqueeze(0)         # (1, k, H, W) or (1, H, W)

    # 4) compute GED on these hard slices
    # flatten extra dims so deg sees (M, k*H*W)
    flat_samps = hard_samps.view(M, -1)
    flat_ref   = hard_ref.view(-1)
    # pairwise distances
    d_sr = torch.stack([distance_fn(flat_samps[i], flat_ref) for i in range(M)]).mean()
    d_ss = torch.stack([
        distance_fn(flat_samps[i], flat_samps[j])
        for i, j in itertools.product(range(M), repeat=2)
    ]).mean()
    return 2 * d_sr - d_ss