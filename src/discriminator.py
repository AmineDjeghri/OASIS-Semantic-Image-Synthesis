from collections import OrderedDict
from collections import namedtuple
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from modules import ResnetBlock

class Discriminator(nn.Module):
    def __init__(self, out_dim, down_hidden_dims, up_hidden_dims):
        super().__init__()
        self.resnet_blocks_down = nn.ModuleList()
        self.resnet_blocks_up = nn.ModuleList()
        for hin, hout in zip(down_hidden_dims[:-1], down_hidden_dims[1:]):
            self.resnet_blocks_down.append(
                ResnetBlock(fin=hin, fout=hout, up=False)
            )
        for hin, hout in zip(up_hidden_dims[:-1], up_hidden_dims[1:]):
            self.resnet_blocks_up.append(
                ResnetBlock(fin=hin*2, fout=hout, up=True)
            )

        self.conv = nn.Conv2d(up_hidden_dims[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x : batch of images (bs, 3, height, width)
        """
        # (1) Resnet Blocks Down
        downsized = []
        for resblock in self.resnet_blocks_down:
            x = resblock(x)
            downsized.append(x)
        
        # (2) Resnet Blocks => Up
        for n, resblock in enumerate(self.resnet_blocks_up):
            if n > 0:
                x = torch.cat([x, downsized[-(n+1)]], 1)
            x = resblock(x)
        
        # (3) Final conv layer
        x = self.conv(x)

        return x