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

        self.fourier = FourierFeatureEncoding(8)

        self.condition = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, cond_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.initial = nn.Conv3d(3 + cond_dim, 64, 3, padding=1)

        self.res32 = nn.Sequential(
            ResNetBlock3D(64, 64),
            ResNetBlock3D(64, 64),
        )

        # 32 → 64
        self.up1 = nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1)

        self.res64 = nn.Sequential(
            ResNetBlock3D(64, 64),
            ResNetBlock3D(64, 64),
        )

        # 64 → 128
        self.up2 = nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1)

        self.res128 = nn.Sequential(
            ResNetBlock3D(64, 64),
            ResNetBlock3D(64, 64),
        )

        self.final = nn.Conv3d(64, 3, 1)

    def forward(self, x, s):

        B, _, D, H, W = x.shape

        # Save input for residual learning
        x_input = x

        # Conditioning
        cond_feat = self.fourier(s.view(B, 1))
        cond_feat = self.condition(cond_feat)
        cond_exp = cond_feat.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)

        x = torch.cat([x, cond_exp], dim=1)
        x = self.initial(x)

        # 32³ processing
        x = self.res32(x)

        # 32 → 64
        x = self.up1(x)
        x = self.res64(x)

        # 64 → 128
        x = self.up2(x)
        x = self.res128(x)

        residual = self.final(x)

        # Upsample LR input for residual base
        base = F.interpolate(
            x_input,
            scale_factor=4,
            mode="trilinear",
            align_corners=False
        )

        return base + residual
