import argparse
import logging
from PIL import Image
import time

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataset import ADEDataset
from generator import Generator
from utils import *

# Semantic Segmentation Library
import os
import sys
sys.path.append(str(Path(__file__).parent.parent / "pretrained"))
from load_pretrained import load_pretrained

to_normalize = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

def per_class_IOU(yhat, annot, c):
    intersection = (annot == c) & (yhat == c)
    card_intersection = intersection.sum()
    union = (annot == c) | (yhat == c)
    card_union = union.sum()
    return card_intersection / card_union

def compute_mIOU_score(opt, checkpoint, model):
    val_dataset = ADEDataset(
        path_to_images=Path(opt.data.path) / "images"/ "validation",
        path_to_annotations=Path(opt.data.path) / "annotations" / "validation",
        load_shape=(opt.data.load_height, opt.data.load_width)
    )

    loader = DataLoader(val_dataset, opt.batch_size, True, drop_last=True)

    G = Generator(
        latent_dim=opt.latent_dim,
        n_classes=opt.data.nb_classes,
        hidden_dims=opt.generator_params.hidden_dims
    ).to(device)

    ckpt = torch.load(checkpoint)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()
    z_shape = (opt.batch_size, opt.latent_dim, opt.data.load_height, opt.data.load_width)
    z_fixed = torch.randn((opt.batch_size, opt.latent_dim, 1, 1), device=device).expand(z_shape)
    
    mIOU = 0
    for batch in loader:
        _, annot = batch
        annot = annot.to(device)
        label = one_hot_encode(annot, opt.data.nb_classes)

        # Generate Images
        with torch.no_grad():
            generated = G(z_fixed, label.to(torch.float)) + 1
            generated = to_normalize(generated)
            
            preds = model(feed_dict={"img_data": generated, "seg_label": annot},
                            segSize=generated.size(-1))
            yhat = torch.argmax(preds, 1).squeeze()

            count = 0
            classes = torch.unique(torch.cat([yhat.flatten(), annot.flatten()]))
            ious = np.zeros(classes.size(0))
            for i,c in enumerate(classes):
                ious[i] = per_class_IOU(yhat, annot, c)
            mIOU += ious.mean()
    return mIOU / len(loader)

if __name__ == "__main__":
    # Logger and parser
    logger = logging.Logger("train")
    parser = argparse.ArgumentParser(description='Train OASIS model.')
    parser.add_argument('config', help='path to the `config.yml` file')
    parser.add_argument('--checkpoint', required=True, help='path to checkpoint from which to resume training')
    parser.add_argument('--segmentation_model', required=True, help='path to segmentation module')

    args = parser.parse_args()
    opt = load_yaml(args.config)
    device = torch.device(opt.device)
    torch.manual_seed(opt.seed)

    model = load_pretrained(args.segmentation_model)
    print("mIOU: ", compute_mIOU_score(opt, args.checkpoint, model))
