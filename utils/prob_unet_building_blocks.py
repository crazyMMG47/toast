#### This module contains the building blocks of the probabilistic unet model. ###$
### Created by Hailin Liang. Jul 1st, 2025. 
# Reference to the original probabilistic unet paper github link: 
# https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

# importing
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from monai.networks.nets import UNet
from monai.networks.layers.simplelayers import SkipConnection
from torch.cuda.amp import autocast,GradScaler
from monai.metrics import DiceMetric


### helper functions to extract the unet blocks 
# these functions will correctly parse the MONAI Unet structures 
# The MONAI UNet is constructed recursively, with each level containing an encoder block,
# a skipconnection, and a decoder block. 
# The below two helper functions will extract the encoder and decoder blocks 
# according to the MONAI's strategy in constructing encoders and decoders. 

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


def extract_unet_decoder_blocks(unet: UNet) -> List[nn.Module]:
    """
    Recursively traverses a MONAI UNet to extract the decoder (upsampling) blocks
    in bottom-up order (from deepest to shallowest).
    """
    blocks = []
    def _traverse(module):
        # A MONAI UNet level is a Sequential of [encoder_block, SkipConnection, decoder_block]
        if isinstance(module, nn.Sequential) and isinstance(module[1], SkipConnection):
            # Recurse to the deepest part of the network first.
            _traverse(module[1].submodule)
            # Add the decoder block (up_layer) of the current level on the way back up.
            blocks.append(module[2])
    
    _traverse(unet.model)
    return blocks


# prior net
'''
Dimension of the prior net explained:
Input (B, 1, H, W, D) where B = batch size, 1 = grey image, D = depth (3d data)
-> UNet encoder (downsampling via stride of 2)
-> Bottleneck block (no downsampling)
-> Global average pooling (B, 256, 10, 10, 5) -> (B, 256, 1, 1, 1)
-> (B, 2 * latent_dim, 1, 1, 1)
-> Flatten to (B, 2 * latent_dim)
-> Chunk -> mu, logvar -> (B, latent_dim) each 
'''

class PriorNet(nn.Module):
    """
    Prior network p(z|x) reusing MONAI's UNet encoder
    
    The prior net is part of a variational model and it will estimate prior distribution p(z|x) over latent variable z. 
    """
    # initialize the PriorNet
    # input_channels: number of input channels (e.g., 1 for grayscale images)
    # latent_dim: dimension of the latent space
    # spatial_dims: number of spatial dimensions (2 or 3)
    # feature_channels: tuple of feature channels for each level in the UNet
    # num_res_units: number of residual units in each block
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        spatial_dims: int = 3, # we are using "3" because we have 3D data
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256), # unless specified, these will be used 
        num_res_units: int = 2, 
        act="PRELU", # parametric ReLU activation function
        norm="INSTANCE",
        dropout: float = 0.2, # TODO: adjust this accordingly! 
        # set for regularization purposes, add stochasticity to the model
        # this dropout only happens in training, not in inference
    ):
        
        super().__init__()
        self.latent_dim = latent_dim
        
        # Appending the last feature channel to the channels tuple
        # e.g. if the original feature channels are: feature_channels = (32, 64, 128, 256)
        # then the channels will be: channels = (32, 64, 128, 256, 256)
        # Adding an extra depth without increasing the no. channels BECAUSE the bottom layer is the bottleneck layer 
        channels = tuple(feature_channels) + (feature_channels[-1],)
        
        # we will not apply the stride to the bottleneck layer
        # And we are using stride of 2 for all other layers --> halving the dimension at each level 
        strides = tuple([2] * len(feature_channels))
        
        # initialize the MONAI's unet with the specified parameters 
        temp_unet = UNet(
            spatial_dims=spatial_dims, in_channels=input_channels, out_channels=1,
            channels=channels, strides=strides, num_res_units=num_res_units,
            act=act, norm=norm, dropout=dropout,
        )
        
        # UNET encoder
        self.encoder = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        
        # nn.Conv3d
        ConvNd = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        # global average pooling (i.e. (B, 256, 10, 10, 5) → (B, 256, 1, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        
        # 1*1*1 convolution, maps pooled features to a vector of size 2 * latent_dim
        # double the size of the tensor shape because we needs to fit both mu and log var in one forward pass
        self.latent_head = ConvNd(feature_channels[-1], 2 * latent_dim, kernel_size=1, bias=True)
        # initialize the bias term of the final conv layer to zero 
        # we are predicting the mean µ and log-var 
        # zeroing out gives µ = 0 and log var = 0 --> sigma = 1 
        # which means the prior distribution is z ~ N (0, 1)
        # This give the NN a neutral starting point 
        nn.init.zeros_(self.latent_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method returns the mean and log-variance of a Gaussian latent distribution inferred from input x.
        
        x will take in the shape (B, C, H, W, D) which is a 3d image batch. 
        
        
        """
        feats = x
        # apply all encoder blocks to the input "x" --> extract encoder block from temp unet 
        for blk in self.encoder:
            feats = blk(feats)
        # apply global average pooling to the features 
        # latent head (1*1 convolution) + flatten 
        stats = self.latent_head(self.global_pool(feats)).flatten(1)
        # return two tensors:
        # 1. (B, mu)
        # 2. (B, latet_dim)
        # torch.chunk will split the pices into two 
        return torch.chunk(stats, 2, dim=1)
    

"""
Posterior Net: 
"""
class PosteriorNet(nn.Module):
    """
    Posterior network q(z|x, y) reusing MONAI's UNet encoder.
    """

    def __init__(
        self,
        image_channels: int,
        mask_channels: int,
        latent_dim: int,
        spatial_dims: int = 3,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256),
        num_res_units: int = 2,
        act="PRELU",
        norm="INSTANCE",
        dropout: float = 0.2, # TODO: adjust this dropout rate if needed 
    ):
        
        super().__init__()
        # this is different from the prior net
        # the posterior net takes in both the image and the mask as input
        # image_channels: number of channels in the input image (e.g., 1 for grayscale images)
        # mask_channels: number of channels in the input mask (e.g., 1 for binary masks)
        in_ch = image_channels + mask_channels
        # add one more last channel to the feature channels because the last layer is the bottleneck layer (similar to the prior net)
        channels = tuple(feature_channels) + (feature_channels[-1],)
        strides = tuple([2] * len(feature_channels))
        temp_unet = UNet(
            spatial_dims=spatial_dims, in_channels=in_ch, out_channels=1,
            channels=channels, strides=strides, num_res_units=num_res_units,
            act=act, norm=norm, dropout=dropout,
        )
        self.encoder = nn.ModuleList(extract_unet_encoder_blocks(temp_unet))
        ConvNd = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.latent_head = ConvNd(feature_channels[-1], 2 * latent_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.latent_head.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, mask], dim=1)
        feats = inp
        for blk in self.encoder:
            feats = blk(feats)
        stats = self.latent_head(self.global_pool(feats)).flatten(1)
        # return two tensors:
        # 1. (B, mu)
        # 2. (B, latet_dim)
        # torch.chunk will split the pices into two
        return torch.chunk(stats, 2, dim=1)
    
    
"""
Fcomb is the building block that can combines the latent z from posterior net to the last layer of the UNet decoder.

Through experiments, we found that the dropout rate can impact a lot on KL. (prevent KL collapse)

1. through the posterior net, we can sample latent vector from mu and logvar
2. we can't directly concatenate z and the feature because they have size mismatch 
3. in order to put them together, we 
"""
class Fcomb(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int,
                 seg_out_channels: int, spatial_dims: int = 3,
                 hidden_ch: int = 64,     # you can reuse feature_channels[0]
                 n_layers: int = 3,
                 drop_p: float = 0.4, # ← dropout prob
                 inject_latent: bool = True):   # add a flag here 
        """
        note:
        hidden_ch: number of intermediate channelsin the 1*1 convolution layers inside Fcomb. 
        
        
        """
        super().__init__()
        self.inject_latent = inject_latent
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
    
        Drop = nn.Dropout3d if spatial_dims == 3 else nn.Dropout2d

        if inject_latent:
            in_feats = in_ch + latent_dim  # after concat
            layers = []
            for _ in range(n_layers - 1):
                layers += [
                    Conv(in_feats, hidden_ch, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    Drop(p=drop_p)                 # TODO: mofify the dropout rate ← dropout after each 1×1 conv
                ]
                in_feats = hidden_ch

            layers += [Conv(hidden_ch, seg_out_channels, kernel_size=1, bias=True)]
            # the fcomb contains:
            # 1. a series of 1x1 convolutions with ReLU activation and dropout
            # 2. a final 1x1 convolution that maps to the output channels (seg_out_channels)
            #    which is the number of classes in the segmentation task
            # The final output will have the shape (B, seg_out_channels, H, W, D) or (B, seg_out_channels, H, W)
            # depending on the spatial dimensions.
            self.fcomb = nn.Sequential(*layers)
            
        else:
            # deterministic 1x1 conv, no z used
            self.fcomb = Conv(in_ch, seg_out_channels, kernel_size=1, bias=True)
            
    def forward(self, feat, z):
        """
        Args:
        -- feat: tensor of shape (B, C, H, W, D) or (B, C, H, W)
        -- z: tensor of shape (B, latent_dim) or (B, latent_dim, 1, 1, 1) or (B, latent_dim, 1, 1)
        """
        # TODO: uncomment the print statements for debugging
        # print(f"[Fcomb] Input feat shape: {feat.shape}")
        # Input feat shape: torch.Size([8, 32, 128, 128, 64])
        # print(f"[Fcomb] Latent z shape before reshape: {z.shape}")
        #  Latent z shape before reshape: torch.Size([8, 32])
        if not self.inject_latent:
            return self.fcomb(feat)  # if not injecting latent, just return the feature through 1x1 conv
        
        # tile z to match satial dim 
        # tile z to H×W (or D×H×W) and concat
        while z.dim() < feat.dim():
            # z shape : (B, latent_dim)
            # below expands z to have the same shape as the feature (B, latent_dim, 1, 1, 1) 
            z = z.unsqueeze(-1)
        
        # now they have the same shape 
        # we need to expand across the spatial dimension
        # for example: initial feat = (2,32,4,4,4) but z = (2,8) which 8 being the latent dimension
        # we need to broadcast z across spatial dimensions  and it will become (2,8,4,4,4)
        z = z.expand_as(feat[:, :z.size(1)])    # broadcast over spatial dims
        # x.shape → (2, 40, 4, 4, 4)   # 32 from feat + 8 from z
        x = torch.cat([feat, z], dim=1)
        return self.fcomb(x)
    
