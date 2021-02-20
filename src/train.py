import argparse
import logging
import time

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

if __name__ == "__main__":
    args = parser.parse_args()
    opt = load_yaml(args.config)
    device = torch.device(opt.device)
    torch.manual_seed(opt.seed)

    train_dataset = ADEDataset(
        path_to_images=Path(opt.data.path) / "images"/ "training",
        path_to_annotations=Path(opt.data.path) / "annotations" / "training",
        load_shape=(opt.data.load_height, opt.data.load_width)
    )

    train_loader = DataLoader(train_dataset, opt.batch_size, True, drop_last=True)
    class_weights = get_weights(opt.data.class_weights, device, opt)

    G = Generator(
        latent_dim=opt.latent_dim,
        n_classes=opt.data.nb_classes,
        hidden_dims=opt.generator_params.hidden_dims #[1024, 512, 256, 128, 64]
    ).to(device)

    D = Discriminator(
        out_dim=opt.data.nb_classes+1,
        down_hidden_dims=opt.discriminator_params.down_dims,
        up_hidden_dims=opt.discriminator_params.up_dims
    ).to(device)

    tb = SummaryWriter(opt.tb_folder)    
    unweighted_CE = nn.CrossEntropyLoss()
    weighted_CE = nn.CrossEntropyLoss(class_weights)
    optim_D = optim.Adam(D.parameters(), lr=opt.optim.lr_D, betas=(opt.optim.beta1, opt.optim.beta2))
    optim_G = optim.Adam(G.parameters(), lr=opt.optim.lr_G, betas=(opt.optim.beta1, opt.optim.beta2))
    ema_G = ExponentialMovingAverage(G.parameters(), decay=opt.EMA_decay)

    print(f"Num params: \n Generator:{get_n_params(G, dimensions=False)}\n Discriminator: {get_n_params(D, dimensions=False)}")

    z_shape = (opt.nb_examples, opt.latent_dim, opt.data.load_height, opt.data.load_width)
    z_fixed = torch.randn((opt.nb_examples, opt.latent_dim, 1, 1), device=device).expand(z_shape)
    
    for epoch in range(opt.epochs):
        
        t = 0
        start = time.time()
        epoch_G_loss = 0
        epoch_D_loss = 0
        
        for batch in train_loader:
            img, annot = batch
            img = img.to(device)
            annot = annot.to(device)

            label_gen = torch.ones(annot.shape, dtype=torch.long, device=device) * (opt.data.nb_classes)

            z_shape = (img.size(0), opt.latent_dim, opt.data.load_height, opt.data.load_width)
            z = torch.randn((img.size(0), opt.latent_dim, 1, 1), device=device).expand(z_shape)
            label = one_hot_encode(annot, opt.data.nb_classes)

            # Optimize Discriminator
            gen = G(z, label.to(torch.float))
            Dx_gen = D(gen)
            Dx_data = D(img)

            mask = sample_mask(annot)
            lmix = labelmix(img, gen, mask)
            Dx_lmix = D(lmix)
            lmix_Dx = labelmix(Dx_data, Dx_gen, mask)
            label_mix_regul = torch.norm(Dx_lmix - lmix_Dx, p=None) ** 2

            loss_D = weighted_CE(Dx_data, annot) + unweighted_CE(Dx_gen, label_gen)
            loss_D += opt.lambda_lm * label_mix_regul
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Optimize Generator
            gen = G(z, label.to(torch.float))
            Dx_gen = D(gen)
            loss_G = weighted_CE(Dx_gen, annot)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
            ema_G.update(G)

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            
            tb.add_scalar('LossGenerator', loss_G.item(), t)
            tb.add_scalar('LossDiscriminator', loss_D.item(), t)

            t+=1

        stop = time.time()
        epoch_G_loss /= len(train_loader)
        epoch_D_loss /= len(train_loader)
        print(f"""Epoch: {epoch} done in {(stop - start) / 60 :.2} min\n 
        Generator Loss: {epoch_G_loss :.4} - Discriminator Loss: {epoch_D_loss :.4}""")
        
        # Visualizing the epoch generation
        examples = G(z_fixed, label[:opt.nb_examples]).detach().cpu()
        visualize_generation(
            examples, 
            annot[:opt.nb_examples, None, :, :].cpu(),
            img[:opt.nb_examples].cpu(),
            tb=tb,
            i=epoch
        )

        # Save if necessary
        if epoch % opt.freq_save == 0:
            logger.info("Saving model ...")
            save_path_checkpoints = Path(opt.checkpoint_path) / f"oasis_{epoch}.pt"
            save_dict = {
                        'D_state_dict': D.state_dict(),
                        'G_state_dict': G.state_dict(),
                        'optimizerD_state_dict': optim_D.state_dict(),
                        'optimizerG_state_dict': optim_G.state_dict(),
                        'epoch': epoch,
                        'last_lossG': epoch_G_loss,
                        'last_lossD': epoch_D_loss
                        }
            torch.save(save_dict, save_path_checkpoints)