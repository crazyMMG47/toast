# models/posterior_net.py
import torch
import torch.nn as nn

class PosteriorNet(nn.Module):
    """Network for modeling the posterior distribution in latent space."""
    
    def __init__(self, input_channels=1, hidden_dims=[32, 64, 128], latent_dim=16, 
                 distribution_type="gaussian"):
        """
        Args:
            input_channels: Number of input image channels
            hidden_dims: List of hidden dimensions
            latent_dim: Dimension of latent space
            distribution_type: Type of distribution (gaussian, mixture, student_t)
        """
        super().__init__()
        self.distribution_type = distribution_type
        self.latent_dim = latent_dim
        
        # Similar structure to PriorNet but potentially with more capacity
        modules = []
        in_channels = input_channels
        
        # Down-sampling
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate output size of encoder
        self.encoder_output_dim = hidden_dims[-1] * (256 // (2 ** len(hidden_dims))) ** 2
        
        # Distribution parameters
        if distribution_type == "gaussian":
            self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
            self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)
        elif distribution_type == "mixture":
            self.n_mixtures = 5  # Number of Gaussian mixtures
            self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim * self.n_mixtures)
            self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim * self.n_mixtures)
            self.fc_weight = nn.Linear(self.encoder_output_dim, self.n_mixtures)
        elif distribution_type == "student_t":
            self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
            self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)
            self.fc_df = nn.Linear(self.encoder_output_dim, latent_dim)  # Degrees of freedom
        
    def forward(self, x):
        """
        Forward pass to get distribution parameters.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary with distribution parameters
        """
        # Encode
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        
        # Get distribution parameters
        if self.distribution_type == "gaussian":
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            return {"mu": mu, "log_var": log_var}
        
        elif self.distribution_type == "mixture":
            mu = self.fc_mu(x).view(-1, self.n_mixtures, self.latent_dim)
            log_var = self.fc_var(x).view(-1, self.n_mixtures, self.latent_dim)
            weights = torch.softmax(self.fc_weight(x), dim=1)
            return {"mu": mu, "log_var": log_var, "weights": weights}
        
        elif self.distribution_type == "student_t":
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            df = torch.nn.functional.softplus(self.fc_df(x)) + 2  # df > 2 for finite variance
            return {"mu": mu, "log_var": log_var, "df": df}