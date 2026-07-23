import torch.nn as nn
import torch.nn.functional as F

##################
# Network Components
##################
class ResNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock3D, self).__init__()
        self.same_channels = (in_channels == out_channels)

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.activation = nn.ReLU(inplace=True)

        if not self.same_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        identity = self.skip_conv(x)
        out = self.conv_block(x)
        out += identity
        return self.activation(out)


class FourierFeatureEncoding(nn.Module):
    def __init__(self, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.arange(num_frequencies) * math.pi

    def forward(self, x):
        """
        x: (B,1) tensor in [-1,1]
        return: (B, 2*num_frequencies) features
        """
        x = x.view(-1, 1)
        x_proj = x * self.freq_bands.to(x.device)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ConditionalGenerator(nn.Module):
    def __init__(self, cond_dim=64):
        super().__init__()
        self.fourier = FourierFeatureEncoding(num_frequencies=8)
        
        # Improved conditioning network
        self.condition = nn.Sequential(
            nn.Linear(2*8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, cond_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder: initial conv + ResNet blocks
        self.enc_initial = nn.Sequential(
            nn.Conv3d(3 + cond_dim, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder path with downsampling
        self.encoder1 = ResNetBlock3D(64, 64)
        
        self.downsample1 = nn.Conv3d(64, 128, 3, stride=2, padding=1)  # -> 32³
        self.encoder2 = ResNetBlock3D(128, 128)
        
        self.downsample2 = nn.Conv3d(128, 256, 3, stride=2, padding=1)  # -> 16³
        self.encoder3 = ResNetBlock3D(256, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResNetBlock3D(256, 256),
            ResNetBlock3D(256, 256)
        )
        
        # Decoder path with upsampling + skip connections
        self.upsample2 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)  # -> 32³
        self.decoder2 = ResNetBlock3D(256, 128)  # 256 = 128 + 128 (skip connection)
        
        self.upsample1 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)  # -> 64³
        self.decoder1 = ResNetBlock3D(128, 64)  # 128 = 64 + 64 (skip connection)
        
        # Final output layer
        self.final = nn.Sequential(
            ResNetBlock3D(64, 64),
            nn.Conv3d(64, 3, 1)
        )

    def forward(self, x, s):
        """
        x: (B, 3, 64, 64, 64) - input at full 64³ resolution
        s: (B, 1) - redshift condition
        """
        B, _, D, H, W = x.shape

        # Create conditional vector
        cond_feat = self.fourier(s.view(B, 1))
        cond_feat = self.condition(cond_feat)
        cond_exp = cond_feat.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)

        # Store input for residual connection
        x_input = x
        
        # Concatenate condition with input
        x = torch.cat([x, cond_exp], dim=1)

        # Encoding path (with skip connections saved)
        x = self.enc_initial(x)         # (B, 64, 64, 64, 64)
        e1 = self.encoder1(x)           # (B, 64, 64, 64, 64) - Save for skip
        
        x = self.downsample1(e1)        # (B, 128, 32, 32, 32)
        e2 = self.encoder2(x)           # (B, 128, 32, 32, 32) - Save for skip
        
        x = self.downsample2(e2)        # (B, 256, 16, 16, 16)
        e3 = self.encoder3(x)           # (B, 256, 16, 16, 16)
        
        # Bottleneck
        x = self.bottleneck(e3)         # (B, 256, 16, 16, 16)
        
        # Decoding path (with skip connections)
        x = self.upsample2(x)           # (B, 128, 32, 32, 32)
        x = torch.cat([x, e2], dim=1)   # (B, 256, 32, 32, 32) - Concatenate skip
        x = self.decoder2(x)            # (B, 128, 32, 32, 32)
        
        x = self.upsample1(x)           # (B, 64, 64, 64, 64)
        x = torch.cat([x, e1], dim=1)   # (B, 128, 64, 64, 64) - Concatenate skip
        x = self.decoder1(x)            # (B, 64, 64, 64, 64)
        
        # Final output
        delta = self.final(x)           # (B, 3, 64, 64, 64)
        
        # Residual connection: predict correction to input
        return x_input + 0.1 * delta
