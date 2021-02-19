import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm

# Simplified Version of the Base Modules from the SPADE architecture implementation: ResnetBlock and SPADEResnetBlock
# Source: https://github.com/NVlabs/SPADE

class SPADE(nn.Module):
    def __init__(self, num_channels_input, num_channels_label, hidden_dim, kernel_size):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(num_channels_input, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(num_channels_label, hidden_dim, kernel_size=kernel_size, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_dim, num_channels_input, kernel_size=kernel_size, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_dim, num_channels_input, kernel_size=kernel_size, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.batch_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, in_channels_label):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, in_channels_label, hidden_dim=128, kernel_size=3)
        self.norm_1 = SPADE(fmiddle, in_channels_label, hidden_dim=128, kernel_size=3)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, in_channels_label, hidden_dim=128, kernel_size=3)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# ResBlock Up or Down from Brock et al's BIGGan-deep architecture
class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, up=True):
        super().__init__()
        self.learned_shortcut = (fin != fout)

        self.convks1_1 = spectral_norm(nn.Conv2d(fin, fout, 1))
        self.convks1_2 = spectral_norm(nn.Conv2d(fout, fout, 1))
        self.convks3_1 = spectral_norm(nn.Conv2d(fout, fout, 3, padding=1))
        self.convks3_2 = spectral_norm(nn.Conv2d(fout, fout, 3, padding=1))

        self.convks1_shortcut = spectral_norm(nn.Conv2d(fin, fout, 1))

        self.sample_op = nn.Upsample(scale_factor=2) if up else nn.AvgPool2d(2) 
        self.actvn = nn.LeakyReLU(1e-2)

    def forward(self, x):

        st = self.shortcut(x)
        
        x = self.actvn(self.convks1_1(self.actvn(x)))
        x = self.actvn(self.convks3_1(x))
        x = self.actvn(self.convks3_2(x))
        x = self.sample_op(x)
        x = self.convks1_2(x)

        x = x + st
        return x

    def shortcut(self, x):
        st = self.sample_op(x)
        if self.learned_shortcut:
            st = self.convks1_shortcut(st)
        return st