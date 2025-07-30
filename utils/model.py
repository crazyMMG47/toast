# We have a class ModelHome that flexibly implements the Probabilistic U-Net architecture.
# You can also turn off the latent vector injection mode to use it as a deterministic U-Net.

# just in case, we import the necessary libraries 
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from monai.networks.nets import UNet
from monai.networks.layers.simplelayers import SkipConnection
from torch.cuda.amp import autocast,GradScaler
from monai.metrics import DiceMetric
from utils.prob_unet_building_blocks import PriorNet, PosteriorNet, Fcomb
from utils.prob_unet_building_blocks import * # being lazzzy
import torch.nn.functional as F


class ModelHome(nn.Module):
    """
    Implements the Probabilistic U-Net architecture where the latent vector `z`
    is injected at a single point after the main U-Net decoder.
    
    Reparameterization trick is used to sample `z` from the posterior distribution.
    """
    def __init__(self,
                 image_channels: int,
                 mask_channels: int,
                 latent_dim: int,
                 feature_channels: Tuple[int,...],
                 num_res_units: int,
                 act="PRELU",
                 norm="INSTANCE",
                 dropout=0.2,
                 spatial_dims=3, # we're doing 3D here
                 seg_out_channels=1,
                 inject_latent: bool = True):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.inject_latent = inject_latent

        # initialize the UNet backbone
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=image_channels,
            out_channels=feature_channels[0],
            channels=feature_channels,
            strides=tuple([2] * len(feature_channels)),
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        
        self.fcomb = Fcomb(
            in_ch=feature_channels[0],
            latent_dim=latent_dim,
            seg_out_channels=seg_out_channels,
            spatial_dims=spatial_dims,
            inject_latent=inject_latent,
        )

        # Filter if latent vector injection is enabled
        if inject_latent:
            print("Probabilistic U-Net with latent vector injection enabled.PriorNet and PosteriorNet will be used.")
            
            self.prior_net = PriorNet(
                input_channels=image_channels, latent_dim=latent_dim, spatial_dims=spatial_dims,
                feature_channels=feature_channels, num_res_units=num_res_units,
                act=act, norm=norm, dropout=dropout)
            self.post_net = PosteriorNet(
                image_channels=image_channels, mask_channels=mask_channels, latent_dim=latent_dim,
                spatial_dims=spatial_dims, feature_channels=feature_channels,
                num_res_units=num_res_units, act=act, norm=norm, dropout=dropout)

       
    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None, 
                sample_z: torch.Tensor = None):
        
        # vanilla unet 
        if not self.inject_latent:
            # feature extraction 
            feat = self.unet(x)
            # by passing None for z, it simply runs a few conv layers on feat to 
            # produce the final layers of logits 
            logits = self.fcomb(feat, None)
            return logits # raw segmentation logits 
        
        if self.training:
            assert mask is not None, "Mask must be provided for training."

            # Get prior and posterior distributions
            mu_p, logvar_p = self.prior_net(x)
            mu_q, logvar_q = self.post_net(x, mask)

            # Reparameterization trick for q(z|x,y)
            std_q = torch.exp(0.5 * logvar_q)
            z = mu_q + std_q * torch.randn_like(std_q)
            
        else:
            # Inference: sample from prior unless a z is provided
            mu_p, logvar_p = self.prior_net(x)
            if sample_z is None:
                std_p = torch.exp(0.5 * logvar_p)
                z = mu_p + std_p * torch.randn_like(std_p)
            else:
                z = sample_z
            mu_q = logvar_q = None # No posterior at test time

        # UNet backbone → Fcomb
        feat   = self.unet(x)
        logits = self.fcomb(feat, z)

        if self.training:
            # return both posterior and prior distribution's mu and logvar 
            return logits, (mu_p, logvar_p), (mu_q, logvar_q)
        else:
            # During inference/validation, return ONLY the segmentation logits
            return logits