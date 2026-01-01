from torchvision.models import ResNet18_Weights
from torchvision import transforms

def get_transforms(mode="weak", is_ms=False):
    if is_ms:
        base_transform = transforms.Compose([])
    else:
        weights = ResNet18_Weights.DEFAULT
        base_transform = weights.transforms()

    if mode == "weak":
        aug_layers = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),   
        ])
        
    elif mode == "strong":
        if is_ms:
            raise ValueError("Strong augmentation is disabled for MS data.")
        
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
    else:
        print("No transform applied.")
        return None

    return transforms.Compose([aug_layers, base_transform])