# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticSegmentationLoss(nn.Module):
    """Loss function for probabilistic segmentation model."""
    
    def __init__(self, kl_weight=0.1):
        """
        Args:
            kl_weight: Weight for KL divergence term
        """
        super().__init__()
        self.kl_weight = kl_weight
        
    def forward(self, outputs, targets):
        """
        Compute loss.
        
        Args:
            outputs: Dictionary with model outputs
            targets: Target segmentation masks
            
        Returns:
            Total loss and dictionary with loss components
        """
        # Segmentation loss (cross-entropy)
        seg_loss = F.cross_entropy(outputs["logits"], targets)
        
        # KL divergence
        if "mu" in outputs["prior_params"] and "mu" in outputs["posterior_params"]:
            # Gaussian case
            kl_loss = self.kl_divergence_gaussian(
                outputs["posterior_params"]["mu"],
                outputs["posterior_params"]["log_var"],
                outputs["prior_params"]["mu"],
                outputs["prior_params"]["log_var"]
            )
        else:
            # TODO: Implement KL for other distributions
            kl_loss = torch.tensor(0.0, device=seg_loss.device)
        
        # Total loss
        total_loss = seg_loss + self.kl_weight * kl_loss.mean()
        
        return total_loss, {
            "seg_loss": seg_loss.item(),
            "kl_loss": kl_loss.mean().item(),
            "total_loss": total_loss.item()
        }
        
    def kl_divergence_gaussian(self, mu_q, log_var_q, mu_p, log_var_p):
        """
        KL divergence between two Gaussian distributions.
        
        Args:
            mu_q, log_var_q: Parameters of q distribution
            mu_p, log_var_p: Parameters of p distribution
            
        Returns:
            KL divergence per batch element
        """
        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        
        kl = 0.5 * torch.sum(
            log_var_p - log_var_q + (var_q + (mu_q - mu_p)**2) / var_p - 1,
            dim=1
        )
        return kl