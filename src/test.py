from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from dataset import ADEDataset
from discriminator import Discriminator
from generator import Generator
from utils import *

def test_one_hot(annot):
    label = one_hot_encode(annot, opt.nb_classes)
    decoded = one_hot_decode(label)
    assert (annot==decoded).to(torch.float).mean() == 1.0
    assert label.shape == torch.Size([32, 151, 256, 256])

def test_labelmix(img1, img2, labelmap):
    """
    Test Code:
    With img and gen associated to the same labelmap
    """
    mask = sample_mask(labelmap)
    lmix = labelmix(img1, img2, mask)
    for i1, i2, m, l in zip(img1, img2, mask, lmix):
        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax = ax.flatten()
        ax[0].imshow((i1.permute(1,2,0).cpu()+1)/2)
        ax[1].imshow((i2.permute(1,2,0).detach().cpu()+1)/2)
        ax[2].imshow(m.cpu(), cmap="gray")
        ax[3].imshow((l.permute(1,2,0).detach().cpu()+1)/2)
        plt.show()

def test_dataset():
    opt = load_yaml("config.yml")
    train_dataset = ADEDataset(
            path_to_images=Path(opt.data.path) / "images"/ "training",
            path_to_annotations=Path(opt.data.path) / "annotations" / "training",
            load_shape=(opt.data.load_height, opt.data.load_width)
        )
    train_loader = DataLoader(train_dataset, opt.batch_size, True, drop_last=True)
    for a, b in train_loader:
        print(a.shape, b.shape)