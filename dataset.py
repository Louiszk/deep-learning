from splits import create_splits
from transforms import get_transforms
from ms_statistics import get_ms_statistics
from settings import datasets_root, split_size
from torch.utils.data import Dataset
from skimage.io import imread
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import os

class EuroSAT(Dataset):
    def __init__(self, dataset_type: str, split: str, transform_type: str ="val"):
        self.dataset_type = dataset_type
        self.split = split
        self.is_ms = "MS" in dataset_type
        self.transform = get_transforms(transform_type, is_ms=self.is_ms)

        self.dataset_path = os.path.join(datasets_root, dataset_type)
        self.dataset_splits = create_splits(dataset_type, split_size)
        self.filenames_per_class = self.dataset_splits[split]
        self.selected_bands = [1, 2, 3, 4, 7, 12]
        
        self.samples = []
        self.classes = sorted(self.filenames_per_class.keys())
        
        for class_name, file_paths in self.filenames_per_class.items():
            label = self.classes.index(class_name)
            for rel_path in file_paths:
                full_path = os.path.join(self.dataset_path, rel_path)
                self.samples.append((full_path, label))

        if self.is_ms:
            self.ms_mean, self.ms_std = get_ms_statistics(self.dataset_splits)
            
            # Add dimensions
            self.ms_mean = self.ms_mean[:, None, None]
            self.ms_std = self.ms_std[:, None, None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        if not self.is_ms:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            img = imread(path)
            img = img.astype(np.float32) / 65535.0
            img = img[:, :, self.selected_bands]

            # switch channel order
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Standardization
            img = (img - self.ms_mean) / self.ms_std

            if self.transform:
                img = self.transform(img)

        return img, label