from settings import datasets_root
from splits import create_splits
from transforms import get_transforms
from torch.utils.data import Dataset
from typing import Tuple
from PIL import Image
import torch
import os

class EuroSAT(Dataset):
    def __init__(self, dataset_type: str, split: str, split_size: tuple =(250, 100, 200), transform_type: str ="val"):
        self.dataset_type = dataset_type
        self.split = split
        self.transform = get_transforms(transform_type)
        self.dataset_path = os.path.join(datasets_root, dataset_type)
        
        self.dataset_splits = create_splits(dataset_type, split_size)
        self.filenames_per_class = self.dataset_splits[split]
        
        # Flatten into (filename, label_idx)
        self.samples = []
        self.classes = sorted(self.filenames_per_class.keys())
        
        for class_name, file_paths in self.filenames_per_class.items():
            label = self.classes.index(class_name)
            for rel_path in file_paths:
                full_path = os.path.join(self.dataset_path, rel_path)
                self.samples.append((full_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        if "_RGB" in self.dataset_type:
            img = Image.open(path).convert("RGB")
        elif "_MS":
            pass
        else:
            raise ValueError("Other dataset types are not available.")

        if self.transform:
            img = self.transform(img)
            
        return img, label



    
