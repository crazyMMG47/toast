# models/probabilistic_segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ProbabilisticSegmentation(nn.Module):
    """Probabilistic segmentation model with latent space modeling."""
    
    def __init__(self, prior_net, posterior_net, unet, latent_dim=16, 
                 distribution_type="gaussian", n_classes=2):
        """
        Args:
            prior_net: Network for modeling prior distribution
            posterior_net: Network for modeling posterior distribution
            unet: U-Net for segmentation
            latent_dim: Dimension of latent space
            distribution_type: Type of distribution (gaussian, mixture, student_t)
            n_classes: Number of segmentation classes
        """
        super().__init__()
        self.prior_net = prior_net
        self.posterior_net = posterior_net
        self.unet = unet
        self.latent_dim = latent_dim
        self.distribution_type = distribution_type
        self.n_classes = n_classes
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for sampling from Gaussian.
        
        Args:
            mu: Mean vector
            log_var: Log variance vector
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def reparameterize_mixture(self, mu, log_var, weights):
        """
        Reparameterization trick for sampling from Gaussian mixture.
        
        Args:
            mu: Mean vectors for each mixture [B, K, D]
            log_var: Log variance vectors for each mixture [B, K, D]
            weights: Mixture weights [B, K]
            
        Returns:
            Sampled latent vector
        """
        # Select mixture component based on weights
        batch_size = weights.shape[0]
        k_indices = torch.multinomial(weights, 1).view(-1)  # [B]
        
        # Extract corresponding mu and log_var
        batch_indices = torch.arange(batch_size, device=mu.device)
        selected_mu = mu[batch_indices, k_indices]  # [B, D]
        selected_log_var = log_var[batch_indices, k_indices]  # [B, D]
        
        # Sample using reparameterization trick
        return self.reparameterize(selected_mu, selected_log_var)
    
    def reparameterize_student_t(self, mu, log_var, df):
        """
        Reparameterization trick for sampling from Student's t-distribution.
        
        Args:
            mu: Mean vector
            log_var: Log variance vector
            df: Degrees of freedom
            
        Returns:
            Sampled latent vector
        """
        # Sample from standard Student's t-distribution
        u = torch.randn_like(mu)
        v = torch.distributions.chi2.Chi2(df).sample()
        t = u * torch.sqrt(df / v)
        
        # Scale and shift
        std = torch.exp(0.5 * log_var)
        return mu + t * std
    
    def kl_divergence_gaussian(self, mu_q, log_var_q, mu_p, log_var_p):
        """
        KL divergence between two Gaussian distributions.
        
        Args:
            mu_q, log_var_q: Parameters of q distribution
            mu_p, log_var_p: Parameters of p distribution
            
        Returns:
            KL divergence
        """
        var_q = torch.exp(log_var_q)
        var_p = torch.exp(log_var_p)
        
        kl = 0.5 * torch.sum(
            log_var_p - log_var_q + (var_q + (mu_q - mu_p)**2) / var_p - 1,
            dim=1
        )
        return kl
    
    def forward(self, x):
        """
        Forward pass of the probabilistic segmentation model.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Dictionary with model outputs
        """
        # Get prior distribution parameters
        prior_params = self.prior_net(x)
        
        # Get posterior distribution parameters
        posterior_params = self.posterior_net(x)
        
        # Sample from posterior during training, prior during inference
        if self.training:
            if self.distribution_type == "gaussian":
                z = self.reparameterize(posterior_params["mu"], posterior_params["log_var"])
            elif self.distribution_type == "mixture":
                z = self.reparameterize_mixture(
                    posterior_params["mu"], 
                    posterior_params["log_var"], 
                    posterior_params["weights"]
                )
            elif self.distribution_type == "student_t":
                z = self.reparameterize_student_t(
                    posterior_params["mu"], 
                    posterior_params["log_var"], 
                    posterior_params["df"]
                )
        else:
            if self.distribution_type == "gaussian":
                z = self.reparameterize(prior_params["mu"], prior_params["log_var"])
            elif self.distribution_type == "mixture":
                z = self.reparameterize_mixture(
                    prior_params["mu"], 
                    prior_params["log_var"], 
                    prior_params["weights"]
                )
            elif self.distribution_type == "student_t":
                z = self.reparameterize_student_t(
                    prior_params["mu"], 
                    prior_params["log_var"], 
                    prior_params["df"]
                )
        
        # Get segmentation output
        logits = self.unet(x, z)
        
        # Return model outputs
        return {
            "logits": logits,
            "z": z,
            "prior_params": prior_params,
            "posterior_params": posterior_params
        }
