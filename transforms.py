from torchvision.models import ResNet18_Weights
from torchvision import transforms
import torch

def get_transforms(mode="weak"):
    weights = ResNet18_Weights.DEFAULT
    base_transform = weights.transforms()

    if mode == "weak":
        aug_layers = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),   
        ])
        
    elif mode == "strong":
        aug_layers = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2,   
                saturation=0.2, 
                hue=0.08        
            )
        ])
        
    elif mode == "val":
        aug_layers = transforms.Compose([])

    return transforms.Compose([aug_layers, base_transform])