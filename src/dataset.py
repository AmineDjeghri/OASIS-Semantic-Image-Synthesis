from PIL import Image

import matplotlib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from utils import Resize

def get_images(folder, ext=".jpg"):
    """
    folder: (Path), path to folder
    """
    if folder.is_dir():
        return sorted(list(folder.glob('*'+ext)))
    else:
        raise ValueError(f"Path {folder} does not point to a valid folder.")


class ADEDataset(Dataset):
    def __init__(self, path_to_images, path_to_annotations, load_shape):
        super().__init__()
        self.images = get_images(path_to_images, ext=".jpg")
        self.annotations = get_images(path_to_annotations, ext=".png")
        self.resize = Resize(load_shape)

        # Validation checks 
        assert len(self.images) == len(self.annotations), "Dataset requires the same number of images and annotations"
        for img, annot in zip(self.images, self.annotations):
            assert img.stem == annot.stem, "List of images and annotations must be provided in the same order"

    def __getitem__(self, i):
        img = torch.tensor(matplotlib.image.imread(self.images[i]))
        # img = self.resize(Image.open(self.images[i]))
        # img = torch.tensor(np.asarray(img))
        # Convert to RGB if necessary
        if img.ndim == 2:
            img = img[:, :, None].expand((img.size(0), img.size(1), 3))
        
        annot = torch.tensor(matplotlib.image.imread(self.annotations[i])*255, dtype=torch.long)
        img = torch.true_divide(self.resize(img.permute(2, 0, 1)), 255) * 2 - 1
        annot = self.resize(annot[None, :, :]).squeeze()
        # annot = self.resize(Image.open(self.annotations[i]))
        # annot = torch.tensor(np.asarray(annot), dtype=torch.long)
        
        # # Translate and scale images
        # img = torch.true_divide(img.permute(2, 0, 1), 255) * 2 - 1
        # annot = annot * 255
        return img, annot

    def __len__(self):
        return len(self.images)

class CityScapesDataset(Dataset):
    def __init__(self, path_to_images, path_to_annotations, load_shape):
        super().__init__()
        self.images = get_images(path_to_images, ext=".png")
        self.annotations = get_images(path_to_annotations, ext=".png")
        self.resize = Resize(load_shape)

        # Validation checks 
        assert len(self.images) == len(self.annotations), "Dataset requires the same number of images and annotations"
        for img, annot in zip(self.images, self.annotations):
            assert img.stem == annot.stem, "List of images and annotations must be provided in the same order"

    def __getitem__(self, i):
        img = torch.tensor(matplotlib.image.imread(self.images[i]))
        # img = self.resize(Image.open(self.images[i]))
        # img = torch.tensor(np.asarray(img))
        # Convert to RGB if necessary
        if img.ndim == 2:
            img = img[:, :, None].expand((img.size(0), img.size(1), 3))
        
        annot = torch.tensor(matplotlib.image.imread(self.annotations[i])*255, dtype=torch.long)
        img = torch.true_divide(self.resize(img.permute(2, 0, 1)), 255) * 2 - 1
        annot = self.resize(annot[None, :, :]).squeeze()
        # annot = self.resize(Image.open(self.annotations[i]))
        # annot = torch.tensor(np.asarray(annot), dtype=torch.long)
        
        # # Translate and scale images
        # img = torch.true_divide(img.permute(2, 0, 1), 255) * 2 - 1
        # annot = annot * 255
        return img, annot

    def __len__(self):
        return len(self.images)