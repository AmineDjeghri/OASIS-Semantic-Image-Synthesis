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
from discriminator import Discriminator
from generator import Generator
from utils import *


# Logger and parser
logger = logging.Logger("train")
parser = argparse.ArgumentParser(description='Train OASIS model.')
parser.add_argument('config', help='path to the `config.yml` file')
parser.add_argument('--checkpoint', required=True, help='path to checkpoint from which to resume training')
parser.add_argument('--destination', required=True, help='path to destination directory for the generated images')
parser.add_argument('--num', type=int, help='number of samples to generate (modulo the batch size),\
                     < number of validation samples.. If None, one per validation sample.')

if __name__ == "__main__":
    args = parser.parse_args()
    opt = load_yaml(args.config)
    device = torch.device(opt.device)
    dest = Path(args.destination)
    if not dest.is_dir():
        raise ValueError(f'arg `destination` must be a valid directory')
    torch.manual_seed(opt.seed)

    val_dataset = ADEDataset(
        path_to_images=Path(opt.data.path) / "images"/ "validation",
        path_to_annotations=Path(opt.data.path) / "annotations" / "validation",
        load_shape=(opt.data.load_height, opt.data.load_width)
    )

    loader = DataLoader(val_dataset, opt.batch_size, True, drop_last=True)

    G = Generator(
        latent_dim=opt.latent_dim,
        n_classes=opt.data.nb_classes,
        hidden_dims=opt.generator_params.hidden_dims #[1024, 512, 256, 128, 64]
    ).to(device)

    ckpt = torch.load(args.checkpoint)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()
    z_shape = (opt.batch_size, opt.latent_dim, opt.data.load_height, opt.data.load_width)
    z_fixed = torch.randn((opt.batch_size, opt.latent_dim, 1, 1), device=device).expand(z_shape)
    
    c = 0
    for batch in loader:
        _, annot = batch
        annot = annot.to(device)
        label = one_hot_encode(annot, opt.data.nb_classes)

        # Generate Images
        if args.num and (c > args.num):
            break
        with torch.no_grad():
            generated = (G(z_fixed, label.to(torch.float)) + 1) / 2 * 255
            generated = generated.cpu().numpy()
            for img in generated:
                to_save = Image.fromarray(img.astype(np.uint8).transpose(1,2,0))
                to_save.save(dest / f"generated_image_{c}.jpg")
                c += 1