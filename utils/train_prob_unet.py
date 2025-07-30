from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from monai.metrics import DiceMetric
import numpy as np

# utils
from utils.metric import recon_loss_fn, kl_divergence
from utils.helper import (
    _sample_diversity,
    TrainingMetrics
)
# import the important function to reweight the KL loss
from utils.metric import SliceKLWeighting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_prob_unet_slice_kl(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 100,
    lr: float = 1e-4,
    beta_final: float = 0.1,
    beta_warmup: int = 20,
    grad_clip: Optional[float] = None,
    # KL and probabilistic options
    use_kl: bool = True,  # Main switch for probabilistic training
    use_slice_kl: bool = True,  # Whether to use slice-level KL reweighting
    worst_slice_percentage: float = 0.1,
    w_hard: float = 2.0,
    w_other: float = 1.0,
    # dice_threshold: float = 0.3,
):
    """
    Train Probabilistic UNet with optional slice-level KL reweighting.
    
    Args:
        model: The probabilistic UNet model (ModelHome)
        train_loader: Training data loader
        val_loader: Validation data loader  
        device: Device to train on
        num_epochs: Number of training epochs
        lr: Learning rate
        beta_final: Final KL weight after warmup
        beta_warmup: Number of epochs for KL warmup
        grad_clip: Gradient clipping value
        use_kl: Whether to use KL loss (enables probabilistic mode)
        use_slice_kl: Whether to use slice-level KL reweighting (only if use_kl=True)
        worst_slice_percentage: Percentage of worst slices to analyze per subject
        w_hard: Weight for hard slices
        w_other: Weight for normal slices  
        dice_threshold: Threshold for determining hard slices
    """
    
    # Setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    # Determine if we're in probabilistic mode
    # Check if model has inject_latent attribute and use_kl is enabled
    model_has_latent = hasattr(model, 'inject_latent') and model.inject_latent
    is_probabilistic = use_kl and model_has_latent
    
    # If use_kl is True but model doesn't support it, warn and fall back
    if use_kl and not model_has_latent:
        print("Warning: use_kl=True but model doesn't have inject_latent=True. Falling back to deterministic mode.")
        is_probabilistic = False
    
    # Initialize slice KL weighting (only if both use_kl and use_slice_kl are True)
    slice_kl_weighter = None
    if is_probabilistic and use_slice_kl:
        slice_kl_weighter = SliceKLWeighting(
            worst_slice_percentage=worst_slice_percentage,
            w_hard=w_hard,
            w_other=w_other
            # dice_threshold=dice_threshold
        )
    
    # Initialize tracking
    best_val_dice = -1.0
    metrics = TrainingMetrics()
    
    # Setup history tracking
    base_keys = ["loss", "dice", "acc"]
    prob_keys = ["kl", "lat_var", "active_dims"] if is_probabilistic else []
    slice_keys = ["w_bar", "kl_vol", "hard_pct"] if (is_probabilistic and use_slice_kl) else []
    
    history = {}
    for split in ['train', 'val']:
        for key in base_keys + prob_keys + slice_keys:
            history[f"{split}_{key}"] = []
    
    if is_probabilistic:
        history["diversity"] = []
    
    # Print training configuration
    mode_str = "Probabilistic" if is_probabilistic else "Deterministic"
    kl_str = "Enabled" if is_probabilistic else "Disabled"
    slice_kl_str = "Enabled" if (is_probabilistic and use_slice_kl) else "Disabled"
    
    print(f"Training mode: {mode_str}")
    print(f"KL Loss: {kl_str}")
    print(f"Slice-level KL: {slice_kl_str}")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        metrics.reset_epoch()
        
        # KL annealing (only relevant if using KL)
        beta = beta_final * min(1.0, epoch / max(beta_warmup, 1)) if is_probabilistic else 0.0
        
        # Slice KL diagnostics accumulators
        epoch_slice_diagnostics = {
            'w_bar': [],
            'kl_vol': [], 
            'hard_pct': []
        }
        
        # Training batches
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                # Forward pass
                if is_probabilistic:
                    # Pass mask for training in probabilistic mode
                    model_output = model(x, y)
                else:
                    # Deterministic mode - no mask needed
                    model_output = model(x)
                
                logits = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
                
                # Reconstruction loss
                loss_recon = recon_loss_fn(logits, y)
                
                # KL loss computation
                loss_kl = torch.zeros((), device=device)  # Default to zero
                
                if is_probabilistic and isinstance(model_output, (list, tuple)):
                    if use_slice_kl and slice_kl_weighter:
                        # Use slice-level weighted KL
                        loss_kl, slice_diag = slice_kl_weighter(model_output, logits, y)
                        
                        # Accumulate diagnostics
                        for key in epoch_slice_diagnostics:
                            if key in slice_diag:
                                epoch_slice_diagnostics[key].append(slice_diag[key])
                    else:
                        # Standard volume-level KL
                        _, (mu_p, logvar_p), (mu_q, logvar_q) = model_output
                        loss_kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
                
                # Total loss with optional KL component
                if is_probabilistic:
                    gamma = 0.5  # KL lower bound
                    total_loss = loss_recon + beta * torch.clamp(loss_kl, min=gamma)
                else:
                    total_loss = loss_recon
            
            # Backward pass
            scaler.scale(total_loss).backward()
            
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            metrics.update_batch(logits.detach(), y, loss_kl.detach(), total_loss.detach())
            
            # Update latent metrics only in probabilistic mode
            if is_probabilistic and isinstance(model_output, (list, tuple)):
                _, _, (mu_q, _) = model_output
                metrics.update_latents(mu_q.detach())
        
        # Finalize epoch metrics
        train_metrics = metrics.finalize(batch_size=x.size(0))
        
        # Store training metrics
        for key in base_keys + prob_keys:
            if key in train_metrics:
                history[f"train_{key}"].append(train_metrics[key])
        
        # Store slice KL diagnostics
        if is_probabilistic and use_slice_kl:
            for key in slice_keys:
                if epoch_slice_diagnostics[key]:
                    history[f"train_{key}"].append(np.mean(epoch_slice_diagnostics[key]))
                else:
                    history[f"train_{key}"].append(0.0)
        
        # Validation
        model.eval()
        val_dice_metric = DiceMetric(include_background=True, reduction="mean")
        val_acc = val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_val in val_loader:
                x_val, y_val = batch_val["image"].to(device), batch_val["label"].to(device)
                
                # Forward pass (no teacher forcing for validation)
                out_val = model(x_val)  # Always inference mode for validation
                logits_val = out_val[0] if isinstance(out_val, (list, tuple)) else out_val
                
                # Metrics
                preds = (torch.sigmoid(logits_val) > 0.5).float()
                val_dice_metric(y_pred=preds, y=y_val)
                val_acc += preds.eq(y_val).float().mean().item()
                val_loss += recon_loss_fn(logits_val, y_val).item()
                val_batches += 1
                
                # Use only one batch for faster validation
                break
        
        # Aggregate validation metrics
        val_dice = val_dice_metric.aggregate().item()
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc / max(val_batches, 1))
        history["val_loss"].append(val_loss / max(val_batches, 1))
        
        # Sample diversity for probabilistic models
        if is_probabilistic:
            history["diversity"].append(_sample_diversity(model, x_val))
        
        # Logging - Format all metrics consistently
        log_msg = f"[{epoch:03d}] trDice={train_metrics['dice']:.4f} | valDice={val_dice:.4f}"
        
        if is_probabilistic:
            kl_val = train_metrics.get('kl', 0.0)
            active_dims = train_metrics.get('active_dims', 0)
            log_msg += f" | KL={kl_val:.3f} | β={beta:.3f} | actDims={active_dims:02d}"
        
        if is_probabilistic and use_slice_kl:
            # Ensure slice KL metrics are always shown, even if zero
            w_bar = history["train_w_bar"][-1] if history["train_w_bar"] else 0.0
            kl_vol = history["train_kl_vol"][-1] if history["train_kl_vol"] else 0.0
            hard_pct = history["train_hard_pct"][-1] if history["train_hard_pct"] else 0.0
            log_msg += f" | w̄={w_bar:.2f} | KLvol={kl_vol:.3f} | hard%={hard_pct:.2f}"
        
        print(log_msg)
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            # Save with appropriate filename based on mode
            filename = "best_probunet_slice_kl.pt" if is_probabilistic else "best_unet_deterministic.pt"
            torch.save(model.state_dict(), filename)
            print(" ✓ New best saved")
    
    print(f"Training completed. Best validation Dice: {best_val_dice:.4f}")
    print(f"Mode used: {'Probabilistic' if is_probabilistic else 'Deterministic'}")
    return history