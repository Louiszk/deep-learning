from settings import project_root, datasets_root
from skimage.io import imread
import numpy as np
import torch
import json
import os

def get_ms_statistics(dataset_splits):
    stats_dir = os.path.join(project_root, "splits")
    stats_path = os.path.join(stats_dir, "ms_stats.json")
    
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return torch.tensor(stats["mean"]), torch.tensor(stats["std"])
    
    print("Calculating mean and std for EuroSAT MS Train.")

    train_files = []
    dataset_path = os.path.join(datasets_root, "EuroSAT_MS")
    
    for _, files in dataset_splits["train"].items():
        for rel_path in files:
            train_files.append(os.path.join(dataset_path, rel_path))
            
    selected_bands = [1, 2, 3, 4, 7, 12]
    
    channels_sum = np.zeros(6)
    channels_sq_sum = np.zeros(6)
    num_pixels = 0
    
    for full_path in train_files:
        img = imread(full_path)
        img = img.astype(np.float32) / 65535.0
        img = img[:, :, selected_bands]
        
        num_pixels += img.shape[0] * img.shape[1]
        channels_sum += np.sum(img, axis=(0, 1))
        channels_sq_sum += np.sum(img ** 2, axis=(0, 1))
        
    mean = channels_sum / num_pixels
    mean_sq = channels_sq_sum / num_pixels

    # variance = E[x^2] - E[x]^2
    std = np.sqrt(mean_sq - (mean ** 2))
    
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    
    with open(stats_path, "w") as f:
        json.dump(stats, f)
        
    print(f"MS Statistics saved.")
    
    return torch.tensor(stats["mean"]), torch.tensor(stats["std"])