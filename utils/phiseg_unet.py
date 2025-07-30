import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from monai.networks.nets import UNet
from monai.networks.blocks import Convolution
from monai.metrics import DiceMetric
import torch.nn.functional as F
from monai.networks.layers.simplelayers import SkipConnection

def extract_unet_encoder_blocks(unet: UNet) -> List[nn.Module]:
    """
    Recursively traverses a MONAI UNet to extract the encoder (downsampling) blocks
    in top-down order (from shallowest to deepest).
    """
    blocks = []
    current_level = unet.model
    # The UNet model is a recursive structure of Sequential modules.
    # Each level contains an encoder block, a SkipConnection, and a decoder block.
    while isinstance(current_level, nn.Sequential) and isinstance(current_level[1], SkipConnection):
        blocks.append(current_level[0])
        current_level = current_level[1].submodule
    # The final level is the bottleneck block.
    blocks.append(current_level)
    return blocks


class PHISegPriorNet(nn.Module):
    """
    PHI-Seg style Prior network p(z|x) using MONAI's UNet encoder
    
    This implements a hierarchical prior network that generates multiple latent variables
    at different resolutions, similar to PHI-Seg's approach.
    """
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256),
        num_res_units: int = 2,
        act="PRELU",
        norm="INSTANCE",
        dropout: float = 0.2,
        latent_levels: int = 5,
        resolution_levels: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.spatial_dims = spatial_dims
        
        # Ensure we have enough feature channels for all levels
        if len(feature_channels) < resolution_levels:
            feature_channels = feature_channels + (feature_channels[-1],) * (resolution_levels - len(feature_channels))
        
        self.feature_channels = feature_channels[:resolution_levels]
        
        # Create strides for downsampling
        strides = tuple([2] * (len(self.feature_channels) - 1))
        
        # Initialize temporary UNet to extract encoder
        temp_unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=input_channels,
            out_channels=1,
            channels=self.feature_channels,
            strides=strides,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        
        # Extract encoder blocks
        self.encoder_blocks = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        
        # Convolution type based on spatial dimensions
        if spatial_dims == 3:
            ConvNd = nn.Conv3d
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            ConvNd = nn.Conv2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Pre-processing convolutions for each resolution level
        self.pre_z_convs = nn.ModuleList()
        for i in range(resolution_levels):
            pre_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            self.pre_z_convs.append(pre_conv)
        
        # TODO: Fix the size mismatch issue here!
        # TODO: NEED A FIX!!!
        # Upsampling convolutions for hierarchical connections
        self.z_ups_convs = nn.ModuleDict()
        
        for i in range(latent_levels):
            # i.e. if latent_levels are 5, i will be 0,1,2,3,4
            for j in range(i + 1):
                # 
                conv_name = f'z_ups_{i}_{j}'
                ups_conv = nn.Sequential(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=latent_dim,
                        out_channels=self.feature_channels[j],
                        kernel_size=3,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                    ),
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=self.feature_channels[j],
                        out_channels=self.feature_channels[j],
                        kernel_size=3,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                    ),
                )
                self.z_ups_convs[conv_name] = ups_conv
        
        # Input processing convolutions for hierarchical levels
        self.z_input_convs = nn.ModuleList()
        for i in range(latent_levels - 1):
            res_level = i + resolution_levels - latent_levels
            in_channels = self.feature_channels[res_level] + self.feature_channels[i]
            
            input_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=self.feature_channels[res_level],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[res_level],
                    out_channels=self.feature_channels[res_level],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            self.z_input_convs.append(input_conv)
        
        # Mean and log-variance heads for each latent level
        self.mu_heads = nn.ModuleList()
        self.logvar_heads = nn.ModuleList()
        
        for i in range(latent_levels):
            res_level = i + resolution_levels - latent_levels
            
            mu_head = ConvNd(
                self.feature_channels[res_level],
                latent_dim,
                kernel_size=1,
                bias=True
            )
            logvar_head = ConvNd(
                self.feature_channels[res_level],
                latent_dim,
                kernel_size=1,
                bias=True
            )
            
            # Initialize biases to zero for neutral prior
            nn.init.zeros_(mu_head.bias)
            nn.init.zeros_(logvar_head.bias)
            
            self.mu_heads.append(mu_head)
            self.logvar_heads.append(logvar_head)
    
    def forward(
        self, 
        x: torch.Tensor, 
        z_list: Optional[List[torch.Tensor]] = None,
        generation_mode: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of PHI-Seg Prior Network
        
        Args:
            x: Input tensor (B, C, H, W, D) for 3D or (B, C, H, W) for 2D
            z_list: List of posterior samples (used during training)
            generation_mode: Whether to use prior samples (True) or posterior samples (False)
            
        Returns:
            Tuple of (z_samples, mu_list, logvar_list)
        """
        # Extract features at different resolutions using encoder
        encoder_features = []
        feats = x
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            feats = encoder_block(feats)
            encoder_features.append(feats)
        
        # Generate pre_z features
        pre_z = []
        for i in range(self.resolution_levels):
            if i < len(encoder_features):
                pre_z_feat = self.pre_z_convs[i](encoder_features[i])
            else:
                # For additional levels, use average pooling
                pre_z_feat = self.avg_pool(encoder_features[-1])
                pre_z_feat = self.pre_z_convs[i](pre_z_feat)
            pre_z.append(pre_z_feat)
        
        # Initialize storage for hierarchical latent variables
        z_samples = [None] * self.latent_levels
        mu_list = [None] * self.latent_levels
        logvar_list = [None] * self.latent_levels
        
        # Matrix to store upsampled z values
        z_ups_mat = []
        for i in range(self.latent_levels):
            z_ups_mat.append([None] * self.latent_levels)
        
        # Generate latent variables from coarse to fine
        for i in reversed(range(self.latent_levels)):
            res_level = i + self.resolution_levels - self.latent_levels
            
            if i == self.latent_levels - 1:
                # Deepest level - use pre_z directly
                z_input = pre_z[res_level]
            else:
                # Hierarchical levels - combine pre_z with upsampled features
                z_below_ups_list = []
                
                for j in reversed(range(i + 1)):
                    # Upsample from level below
                    z_below = z_ups_mat[j + 1][i + 1]
                    z_below_ups = self.upsample(z_below)
                    
                    # Process upsampled features
                    conv_name = f'z_ups_{i+1}_{j}'
                    z_below_ups = self.z_ups_convs[conv_name](z_below_ups)
                    z_ups_mat[j][i + 1] = z_below_ups
                
                # Concatenate pre_z with upsampled features from level below
                z_input = torch.cat([pre_z[res_level], z_ups_mat[i][i + 1]], dim=1)
                z_input = self.z_input_convs[i](z_input)
            
            # Generate mu and logvar
            mu = self.mu_heads[i](z_input)
            logvar = self.logvar_heads[i](z_input)
            
            # Sample z
            if generation_mode or z_list is None:
                # Use prior samples
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                # Use posterior samples during training
                z = z_list[i]
            
            z_samples[i] = z
            mu_list[i] = mu
            logvar_list[i] = logvar
            
            # Store for hierarchical connections
            z_ups_mat[i][i] = z
        
        return z_samples, mu_list, logvar_list
    
    def sample_prior(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Sample from prior distribution p(z|x)
        
        Args:
            x: Input tensor
            
        Returns:
            List of sampled latent variables
        """
        z_samples, _, _ = self.forward(x, generation_mode=True)
        return z_samples
    
    def get_kl_divergence(
        self, 
        mu_list: List[torch.Tensor], 
        logvar_list: List[torch.Tensor],
        kl_weights: Optional[List[float]] = None,
        weighting_strategy: str = "uniform"
    ) -> torch.Tensor:
        """
        Compute weighted KL divergence between prior and standard normal distribution
        
        Args:
            mu_list: List of mean tensors
            logvar_list: List of log-variance tensors
            kl_weights: Optional list of weights for each level
            weighting_strategy: Strategy for weighting KL terms
                - "uniform": Equal weights for all levels
                - "resolution": Weight by spatial resolution (finer levels get higher weight)
                - "inverse_resolution": Weight by inverse resolution (coarser levels get higher weight)
                - "exponential": Exponentially increasing weights for finer levels
                - "custom": Use provided kl_weights
                
        Returns:
            Weighted total KL divergence
        """
        if kl_weights is None:
            if weighting_strategy == "uniform":
                kl_weights = [1.0] * self.latent_levels
            elif weighting_strategy == "resolution":
                # Higher weight for finer resolutions (higher spatial detail)
                kl_weights = [2**i for i in range(self.latent_levels)]
            elif weighting_strategy == "inverse_resolution":
                # Higher weight for coarser resolutions (global structure)
                kl_weights = [2**(self.latent_levels-1-i) for i in range(self.latent_levels)]
            elif weighting_strategy == "exponential":
                # Exponentially increasing weights for finer levels
                kl_weights = [1.5**i for i in range(self.latent_levels)]
            else:
                raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
        
        # Normalize weights to sum to number of levels (maintains scale)
        weight_sum = sum(kl_weights)
        kl_weights = [w * self.latent_levels / weight_sum for w in kl_weights]
        
        kl_total = 0.0
        
        for i, (mu, logvar, weight) in enumerate(zip(mu_list, logvar_list, kl_weights)):
            # KL(q(z|x) || p(z)) where p(z) = N(0, I)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_total += weight * kl.mean()
        
        return kl_total
    
    def get_kl_divergence_per_level(
        self, 
        mu_list: List[torch.Tensor], 
        logvar_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute KL divergence for each level separately
        
        Args:
            mu_list: List of mean tensors
            logvar_list: List of log-variance tensors
            
        Returns:
            List of KL divergences for each level
        """
        kl_per_level = []
        
        for mu, logvar in zip(mu_list, logvar_list):
            # KL(q(z|x) || p(z)) where p(z) = N(0, I)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_per_level.append(kl.mean())
        
        return kl_per_level
    
    def get_kl_divergence_between_distributions(
        self,
        posterior_mu_list: List[torch.Tensor],
        posterior_logvar_list: List[torch.Tensor],
        prior_mu_list: List[torch.Tensor],
        prior_logvar_list: List[torch.Tensor],
        kl_weights: Optional[List[float]] = None,
        weighting_strategy: str = "uniform"
    ) -> torch.Tensor:
        """
        Compute weighted KL divergence between posterior q(z|x,y) and prior p(z|x)
        This is the actual KL term used in PHI-Seg training: KL(q(z|x,y) || p(z|x))
        
        Args:
            posterior_mu_list: List of posterior mean tensors
            posterior_logvar_list: List of posterior log-variance tensors  
            prior_mu_list: List of prior mean tensors
            prior_logvar_list: List of prior log-variance tensors
            kl_weights: Optional list of weights for each level
            weighting_strategy: Strategy for weighting KL terms
                
        Returns:
            Weighted total KL divergence between posterior and prior
        """
        if kl_weights is None:
            if weighting_strategy == "uniform":
                kl_weights = [1.0] * self.latent_levels
            elif weighting_strategy == "resolution":
                kl_weights = [2**i for i in range(self.latent_levels)]
            elif weighting_strategy == "inverse_resolution":
                kl_weights = [2**(self.latent_levels-1-i) for i in range(self.latent_levels)]
            elif weighting_strategy == "exponential":
                kl_weights = [1.5**i for i in range(self.latent_levels)]
            else:
                raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
        
        # Normalize weights
        weight_sum = sum(kl_weights)
        kl_weights = [w * self.latent_levels / weight_sum for w in kl_weights]
        
        kl_total = 0.0
        
        for i, (q_mu, q_logvar, p_mu, p_logvar, weight) in enumerate(
            zip(posterior_mu_list, posterior_logvar_list, prior_mu_list, prior_logvar_list, kl_weights)
        ):
            # KL(q(z|x,y) || p(z|x)) for two Gaussians
            # KL = 0.5 * [log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1]
            kl = 0.5 * torch.sum(
                p_logvar - q_logvar + 
                (q_logvar.exp() + (q_mu - p_mu).pow(2)) / (p_logvar.exp() + 1e-8) - 1,
                dim=1
            )
            kl_total += weight * kl.mean()
        
        return kl_total
    
    
class PHISegPosteriorNet(nn.Module):
    """
    PHI-Seg style Posterior network q(z|x, y) using MONAI's UNet encoder
    
    This implements a hierarchical posterior network that generates multiple latent variables
    at different resolutions, conditioned on both input image and segmentation mask.
    """
    
    def __init__(
        self,
        image_channels: int,
        mask_channels: int,
        latent_dim: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256),
        num_res_units: int = 2,
        act="PRELU",
        norm="INSTANCE",
        dropout: float = 0.2,
        latent_levels: int = 5,
        resolution_levels: int = 6,
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.latent_dim = latent_dim
        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.spatial_dims = spatial_dims
        
        # Input channels: image + mask (similar to x + s_oh in original PHI-Seg)
        in_channels = image_channels + mask_channels
        
        # Ensure we have enough feature channels for all levels
        if len(feature_channels) < resolution_levels:
            feature_channels = feature_channels + (feature_channels[-1],) * (resolution_levels - len(feature_channels))
        
        self.feature_channels = feature_channels[:resolution_levels]
        
        # Create strides for downsampling
        strides = tuple([2] * (len(self.feature_channels) - 1))
        
        # Initialize temporary UNet to extract encoder
        temp_unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=1,
            channels=self.feature_channels,
            strides=strides,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        
        # Extract encoder blocks
        self.encoder_blocks = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        
        # Convolution type and pooling based on spatial dimensions
        if spatial_dims == 3:
            ConvNd = nn.Conv3d
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            ConvNd = nn.Conv2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Pre-processing convolutions for each resolution level
        self.pre_z_convs = nn.ModuleList()
        for i in range(resolution_levels):
            # For the first level, input is concatenated image+mask
            if i == 0:
                input_ch = in_channels
            else:
                input_ch = self.feature_channels[i-1]
                
            pre_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=input_ch,
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            self.pre_z_convs.append(pre_conv)
        
        # Downsampling layers (average pooling)
        if spatial_dims == 3:
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Upsampling convolutions for hierarchical connections
        self.z_ups_convs = nn.ModuleDict()
        for i in range(latent_levels):
            for j in range(i + 1):
                conv_name = f'z_ups_{i+1}_{j+1}'  # Following PHI-Seg naming convention
                ups_conv = nn.Sequential(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=latent_dim,
                        out_channels=latent_dim * self.feature_channels[0] // 32,  # Scale by n0
                        kernel_size=3,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                    ),
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=latent_dim * self.feature_channels[0] // 32,
                        out_channels=latent_dim * self.feature_channels[0] // 32,
                        kernel_size=3,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                    ),
                )
                self.z_ups_convs[conv_name] = ups_conv
        
        # Input processing convolutions for hierarchical levels
        self.z_input_convs = nn.ModuleList()
        for i in range(latent_levels - 1):
            res_level = i + resolution_levels - latent_levels
            in_channels = self.feature_channels[res_level] + latent_dim * self.feature_channels[0] // 32
            
            input_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=self.feature_channels[res_level],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[res_level],
                    out_channels=self.feature_channels[res_level],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            self.z_input_convs.append(input_conv)
        
        # Mean and sigma heads for each latent level
        self.mu_heads = nn.ModuleList()
        self.sigma_heads = nn.ModuleList()
        
        for i in range(latent_levels):
            res_level = i + resolution_levels - latent_levels
            
            mu_head = ConvNd(
                self.feature_channels[res_level],
                latent_dim,
                kernel_size=1,
                bias=True
            )
            # Using softplus activation for sigma (ensures positive values)
            sigma_head = nn.Sequential(
                ConvNd(
                    self.feature_channels[res_level],
                    latent_dim,
                    kernel_size=1,
                    bias=True
                ),
                nn.Softplus()
            )
            
            # Initialize biases to zero
            nn.init.zeros_(mu_head.bias)
            nn.init.zeros_(sigma_head[0].bias)
            
            self.mu_heads.append(mu_head)
            self.sigma_heads.append(sigma_head)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of PHI-Seg Posterior Network
        
        Args:
            x: Input image tensor (B, C, H, W, D) for 3D or (B, C, H, W) for 2D
            mask: Input mask tensor (B, C, H, W, D) for 3D or (B, C, H, W) for 2D
            
        Returns:
            Tuple of (z_samples, mu_list, sigma_list)
        """
        # Concatenate image and mask (similar to x + s_oh - 0.5 in original PHI-Seg)
        # We subtract 0.5 to center the mask values around 0
        mask_centered = mask - 0.5
        inp = torch.cat([x, mask_centered], dim=1)
        
        # Generate pre_z features at different resolutions
        pre_z = []
        net = inp
        
        for i in range(self.resolution_levels):
            if i == 0:
                # First level uses concatenated input
                net = inp
            else:
                # Subsequent levels use downsampled features from previous level
                net = self.downsample(pre_z[i-1])
            
            # Apply pre-processing convolutions
            pre_z_feat = self.pre_z_convs[i](net)
            pre_z.append(pre_z_feat)
        
        # Initialize storage for hierarchical latent variables
        z_samples = [None] * self.latent_levels
        mu_list = [None] * self.latent_levels
        sigma_list = [None] * self.latent_levels
        
        # Matrix to store upsampled z values
        z_ups_mat = []
        for i in range(self.latent_levels):
            z_ups_mat.append([None] * self.latent_levels)
        
        # Generate latent variables from coarse to fine
        for i in reversed(range(self.latent_levels)):
            res_level = i + self.resolution_levels - self.latent_levels
            
            if i == self.latent_levels - 1:
                # Deepest level - use pre_z directly
                z_input = pre_z[res_level]
            else:
                # Hierarchical levels - combine pre_z with upsampled features
                for j in reversed(range(i + 1)):
                    # Upsample from level below
                    z_below = z_ups_mat[j + 1][i + 1]
                    z_below_ups = self.upsample(z_below)
                    
                    # Process upsampled features
                    conv_name = f'z_ups_{i+1}_{j+1}'
                    
                    # TODO: Debug MSG 
                    print(f"Processing conv_name: {conv_name}")
                    print(f"Shape of z_below_ups before conv: {z_below_ups.shape}")
                    print(f"Layer details: {self.z_ups_convs[conv_name]}") # This will show the layer's in/out channels


                    z_below_ups = self.z_ups_convs[conv_name](z_below_ups) # TODO: Fix the Bug Occuring Here 
                    z_ups_mat[j][i + 1] = z_below_ups
                
                # Concatenate pre_z with upsampled features from level below
                z_input = torch.cat([pre_z[res_level], z_ups_mat[i][i + 1]], dim=1)
                z_input = self.z_input_convs[i](z_input)
            
            # Generate mu and sigma
            mu = self.mu_heads[i](z_input)
            sigma = self.sigma_heads[i](z_input)
            
            # Sample z using reparameterization trick
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            
            z_samples[i] = z
            mu_list[i] = mu
            sigma_list[i] = sigma
            
            # Store for hierarchical connections
            z_ups_mat[i][i] = z
        
        return z_samples, mu_list, sigma_list
    
    def get_kl_divergence(
        self, 
        mu_list: List[torch.Tensor], 
        sigma_list: List[torch.Tensor],
        kl_weights: Optional[List[float]] = None,
        weighting_strategy: str = "uniform"
    ) -> torch.Tensor:
        """
        Compute weighted KL divergence between posterior and standard normal distribution
        
        Args:
            mu_list: List of mean tensors
            sigma_list: List of sigma tensors
            kl_weights: Optional list of weights for each level
            weighting_strategy: Strategy for weighting KL terms
                - "uniform": Equal weights for all levels
                - "resolution": Weight by spatial resolution (finer levels get higher weight)
                - "inverse_resolution": Weight by inverse resolution (coarser levels get higher weight)
                - "exponential": Exponentially increasing weights for finer levels
                - "custom": Use provided kl_weights
                
        Returns:
            Weighted total KL divergence
        """
        if kl_weights is None:
            if weighting_strategy == "uniform":
                kl_weights = [1.0] * self.latent_levels
            elif weighting_strategy == "resolution":
                # Higher weight for finer resolutions (higher spatial detail)
                kl_weights = [2**i for i in range(self.latent_levels)]
            elif weighting_strategy == "inverse_resolution":
                # Higher weight for coarser resolutions (global structure)
                kl_weights = [2**(self.latent_levels-1-i) for i in range(self.latent_levels)]
            elif weighting_strategy == "exponential":
                # Exponentially increasing weights for finer levels
                kl_weights = [1.5**i for i in range(self.latent_levels)]
            else:
                raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
        
        # Normalize weights to sum to number of levels (maintains scale)
        weight_sum = sum(kl_weights)
        kl_weights = [w * self.latent_levels / weight_sum for w in kl_weights]
        
        kl_total = 0.0
        
        for i, (mu, sigma, weight) in enumerate(zip(mu_list, sigma_list, kl_weights)):
            # Convert sigma to log-variance for KL computation
            logvar = 2 * torch.log(sigma + 1e-8)  # Add small epsilon for numerical stability
            
            # KL(q(z|x,y) || p(z)) where p(z) = N(0, I)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_total += weight * kl.mean()
        
        return kl_total
    
    def get_kl_divergence_per_level(
        self, 
        mu_list: List[torch.Tensor], 
        sigma_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute KL divergence for each level separately
        
        Args:
            mu_list: List of mean tensors
            sigma_list: List of sigma tensors
            
        Returns:
            List of KL divergences for each level
        """
        kl_per_level = []
        
        for mu, sigma in zip(mu_list, sigma_list):
            # Convert sigma to log-variance for KL computation
            logvar = 2 * torch.log(sigma + 1e-8)
            
            # KL(q(z|x,y) || p(z)) where p(z) = N(0, I)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_per_level.append(kl.mean())
        
        return kl_per_level
    
    def get_log_likelihood(
        self,
        z_samples: List[torch.Tensor],
        mu_list: List[torch.Tensor],
        sigma_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute log-likelihood of z_samples under the posterior distribution
        
        Args:
            z_samples: List of sampled latent variables
            mu_list: List of mean tensors
            sigma_list: List of sigma tensors
            
        Returns:
            Total log-likelihood
        """
        log_likelihood = 0.0
        
        for z, mu, sigma in zip(z_samples, mu_list, sigma_list):
            # Gaussian log-likelihood
            log_prob = -0.5 * torch.log(2 * torch.pi * sigma.pow(2) + 1e-8)
            log_prob -= 0.5 * (z - mu).pow(2) / (sigma.pow(2) + 1e-8)
            log_likelihood += log_prob.sum(dim=1).mean()
        
        return log_likelihood
    
    
class PHISegDecoder(nn.Module):
    """
    PHI-Seg Hierarchical Decoder/Likelihood Network p(y|x,z)
    
    Takes hierarchical latent variables and generates multi-scale segmentation predictions.
    This corresponds to the 'phiseg' function in the original likelihood.py
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256),
        latent_levels: int = 5,
        resolution_levels: int = 6,
        act="PRELU",
        norm="INSTANCE",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.spatial_dims = spatial_dims
        self.latent_levels = latent_levels
        self.resolution_levels = resolution_levels
        self.lvl_diff = resolution_levels - latent_levels
        
        # Ensure we have enough feature channels
        if len(feature_channels) < resolution_levels:
            feature_channels = feature_channels + (feature_channels[-1],) * (resolution_levels - len(feature_channels))
        
        self.feature_channels = feature_channels[:resolution_levels]
        
        # Convolution type based on spatial dimensions
        if spatial_dims == 3:
            ConvNd = nn.Conv3d
            self.upsample_mode = 'trilinear'
        else:
            ConvNd = nn.Conv2d
            self.upsample_mode = 'bilinear'
        
        # Post-processing convolutions for each latent level (z -> post_z)
        self.post_z_convs = nn.ModuleList()
        for i in range(latent_levels):
            post_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=latent_dim,
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            self.post_z_convs.append(post_conv)
        
        # Resolution increase blocks (for upsampling latents to target resolution)
        self.resolution_increase_blocks = nn.ModuleList()
        for i in range(latent_levels):
            blocks = nn.ModuleList()
            # Each latent needs to be upsampled by (resolution_levels - latent_levels) times
            for j in range(self.lvl_diff):
                block = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i],
                    out_channels=self.feature_channels[i],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                )
                blocks.append(block)
            self.resolution_increase_blocks.append(blocks)
        
        # Upstream processing convolutions (for hierarchical combination)
        self.upstream_convs = nn.ModuleList()
        for i in range(latent_levels - 1):
            # Upsampling convolution from level below
            ups_conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.feature_channels[i + 1],
                out_channels=self.feature_channels[i],
                kernel_size=3,
                act=act,
                norm=norm,
                dropout=dropout,
            )
            
            # Combination convolutions after concatenation
            combined_channels = self.feature_channels[i] * 2  # post_z[i] + upsampled
            comb_conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=combined_channels,
                    out_channels=self.feature_channels[i + self.lvl_diff],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=self.feature_channels[i + self.lvl_diff],
                    out_channels=self.feature_channels[i + self.lvl_diff],
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )
            
            self.upstream_convs.append(nn.ModuleDict({
                'ups_conv': ups_conv,
                'comb_conv': comb_conv
            }))
        
        # Final prediction heads for each level
        self.prediction_heads = nn.ModuleList()
        for i in range(latent_levels):
            if i == latent_levels - 1:
                # Deepest level uses its own features
                in_channels = self.feature_channels[i]
            else:
                # Other levels use combined features
                in_channels = self.feature_channels[i + self.lvl_diff]
                
            pred_head = ConvNd(
                in_channels,
                n_classes,
                kernel_size=1,
                bias=True
            )
            self.prediction_heads.append(pred_head)
    
    def increase_resolution(
        self, 
        x: torch.Tensor, 
        times: int, 
        level_idx: int
    ) -> torch.Tensor:
        """
        Increase resolution by upsampling and applying convolutions
        
        Args:
            x: Input tensor
            times: Number of times to upsample (should equal lvl_diff)
            level_idx: Index of the latent level
            
        Returns:
            Upsampled tensor
        """
        net = x
        for i in range(times):
            # Upsample by factor of 2
            net = F.interpolate(
                net, 
                scale_factor=2, 
                mode=self.upsample_mode, 
                align_corners=False
            )
            # Apply convolution
            net = self.resolution_increase_blocks[level_idx][i](net)
        return net
    
    def forward(
        self, 
        z_list: List[torch.Tensor], 
        target_size: Optional[Tuple[int, ...]] = None
    ) -> List[torch.Tensor]:
        """
        Forward pass of PHI-Seg Decoder
        
        Args:
            z_list: List of latent variables from coarse to fine
            target_size: Target spatial size for final predictions
            
        Returns:
            List of segmentation predictions at different scales
        """
        # Step 1: Generate post_z by processing each latent variable
        post_z = []
        for i in range(self.latent_levels):
            # Apply post-processing convolutions
            net = self.post_z_convs[i](z_list[i])
            
            # Increase resolution to match target level
            net = self.increase_resolution(net, self.lvl_diff, i)
            
            post_z.append(net)
        
        # Step 2: Upstream path - hierarchical combination from coarse to fine
        post_c = [None] * self.latent_levels
        
        # Initialize deepest level
        post_c[self.latent_levels - 1] = post_z[self.latent_levels - 1]
        
        # Process from coarse to fine
        for i in reversed(range(self.latent_levels - 1)):
            # Upsample from level below (coarser)
            ups_below = F.interpolate(
                post_c[i + 1],
                scale_factor=2,
                mode=self.upsample_mode,
                align_corners=False
            )
            
            # Apply upsampling convolution
            ups_below = self.upstream_convs[i]['ups_conv'](ups_below)
            
            # Concatenate with current level
            concat = torch.cat([post_z[i], ups_below], dim=1)
            
            # Apply combination convolutions
            post_c[i] = self.upstream_convs[i]['comb_conv'](concat)
        
        # Step 3: Generate predictions at each level
        predictions = []
        for i in range(self.latent_levels):
            # Apply prediction head
            pred = self.prediction_heads[i](post_c[i])
            
            # Resize to target size if specified
            if target_size is not None:
                pred = F.interpolate(
                    pred,
                    size=target_size,
                    mode='nearest'  # Use nearest neighbor for segmentation masks
                )
            
            predictions.append(pred)
        
        return predictions
    
    def sample_predictions(
        self,
        z_list: List[torch.Tensor],
        target_size: Optional[Tuple[int, ...]] = None,
        return_logits: bool = False
    ) -> List[torch.Tensor]:
        """
        Generate segmentation predictions and optionally convert to class predictions
        
        Args:
            z_list: List of latent variables
            target_size: Target spatial size
            return_logits: If False, return softmax probabilities; if True, return raw logits
            
        Returns:
            List of predictions (either logits or probabilities)
        """
        predictions = self.forward(z_list, target_size)
        
        if not return_logits:
            # Convert to probabilities
            predictions = [F.softmax(pred, dim=1) for pred in predictions]
        
        return predictions
    
    def get_multi_scale_loss(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor,
        loss_weights: Optional[List[float]] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute multi-scale reconstruction loss
        
        Args:
            predictions: List of predictions at different scales
            targets: Ground truth segmentation
            loss_weights: Weights for each scale (if None, equal weights)
            loss_fn: Loss function to use (if None, uses CrossEntropyLoss)
            
        Returns:
            Weighted multi-scale loss
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        if loss_weights is None:
            loss_weights = [1.0] * len(predictions)
        
        # Normalize weights
        weight_sum = sum(loss_weights)
        loss_weights = [w / weight_sum for w in loss_weights]
        
        total_loss = 0.0
        
        for pred, weight in zip(predictions, loss_weights):
            # Resize target to match prediction size if needed
            target_resized = targets
            if pred.shape[2:] != targets.shape[2:]:
                target_resized = F.interpolate(
                    targets.float(),
                    size=pred.shape[2:],
                    mode='nearest'
                ).long()
            
            loss = loss_fn(pred, target_resized.squeeze(1))  # Remove channel dim for CE loss
            total_loss += weight * loss
        
        return total_loss


class CompletePHISegModel(nn.Module):
    """
    Complete PHI-Seg Model combining Prior, Posterior, and Decoder networks
    """
    
    def __init__(
        self,
        image_channels: int,
        mask_channels: int,
        latent_dim: int,
        n_classes: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256),
        latent_levels: int = 5,
        resolution_levels: int = 6,
        **kwargs
    ):
        super().__init__()
        
        # Import the networks from our previous artifacts
        # from phiseg_priornet import PHISegPriorNet
        # from phiseg_posteriornet import PHISegPosteriorNet
        
        self.prior_net = PHISegPriorNet(
            input_channels=image_channels,
            latent_dim=latent_dim,
            spatial_dims=spatial_dims,
            feature_channels=feature_channels,
            latent_levels=latent_levels,
            resolution_levels=resolution_levels,
            **kwargs
        )
        
        self.posterior_net = PHISegPosteriorNet(
            image_channels=image_channels,
            mask_channels=mask_channels,
            latent_dim=latent_dim,
            spatial_dims=spatial_dims,
            feature_channels=feature_channels,
            latent_levels=latent_levels,
            resolution_levels=resolution_levels,
            **kwargs
        )
        
        self.decoder = PHISegDecoder(
            latent_dim=latent_dim,
            n_classes=n_classes,
            spatial_dims=spatial_dims,
            feature_channels=feature_channels,
            latent_levels=latent_levels,
            resolution_levels=resolution_levels,
            **kwargs
        )
        
        self.latent_levels = latent_levels
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, ...]] = None,
        use_posterior: bool = True
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Full forward pass
        
        Args:
            x: Input image
            mask: Ground truth mask (for training with posterior)
            target_size: Target size for predictions
            use_posterior: Whether to use posterior (training) or prior (inference)
            
        Returns:
            Tuple of (predictions, kl_loss)
        """
        if use_posterior and mask is not None:
            # Training mode: use posterior
            posterior_z, posterior_mu, posterior_sigma = self.posterior_net(x, mask)
            prior_z, prior_mu, prior_logvar = self.prior_net(x, generation_mode=True)
            
            # Use posterior samples for reconstruction
            z_list = posterior_z
            
            # Compute KL divergence
            posterior_logvar = 2 * torch.log(posterior_sigma + 1e-8)
            kl_loss = self.prior_net.get_kl_divergence_between_distributions(
                posterior_mu, posterior_logvar, prior_mu, prior_logvar,
                weighting_strategy="resolution"
            )
        else:
            # Inference mode: use prior only
            z_list, prior_mu, prior_logvar = self.prior_net(x, generation_mode=True)
            kl_loss = torch.tensor(0.0, device=x.device)
        
        # Generate predictions
        predictions = self.decoder(z_list, target_size)
        
        return predictions, kl_loss
    
    def sample_diverse_predictions(
        self,
        x: torch.Tensor,
        n_samples: int = 5,
        target_size: Optional[Tuple[int, ...]] = None
    ) -> List[List[torch.Tensor]]:
        """
        Generate diverse segmentation samples by sampling from prior
        
        Args:
            x: Input image
            n_samples: Number of diverse samples to generate
            target_size: Target size for predictions
            
        Returns:
            List of prediction lists (one per sample)
        """
        self.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                z_list, _, _ = self.prior_net(x, generation_mode=True)
                predictions = self.decoder(z_list, target_size)
                samples.append(predictions)
        
        return samples