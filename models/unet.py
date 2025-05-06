# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, latent_dim=16, filters=[64, 128, 256, 512], bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.latent_dim = latent_dim

        # Initial feature extraction
        self.inc = DoubleConv(n_channels + latent_dim, filters[0])  # Concatenate latent vector
        
        # Downsampling path
        self.downs = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.downs.append(Down(filters[i], filters[i + 1]))
        
        factor = 2 if bilinear else 1
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            self.ups.append(Up(filters[i], filters[i - 1] // factor, bilinear))
        
        self.outc = OutConv(filters[0], n_classes)

    def forward(self, x, z):
        """
        Forward pass of the U-Net with latent vector integration.
        
        Args:
            x: Input image [B, C, H, W]
            z: Latent vector [B, latent_dim]
            
        Returns:
            Segmentation output
        """
        # Expand latent vector to spatial dimensions
        batch_size, _, h, w = x.shape
        z_expanded = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        # Concatenate input and latent vector
        x = torch.cat([x, z_expanded], dim=1)
        
        # U-Net forward pass
        x1 = self.inc(x)
        
        # Downsampling
        features = [x1]
        for down in self.downs:
            features.append(down(features[-1]))
        
        x = features[-1]
        
        # Upsampling
        for i, up in enumerate(self.ups):
            x = up(x, features[-(i+2)])
        
        # Output layer
        logits = self.outc(x)
        return logits
