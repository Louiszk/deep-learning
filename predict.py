from dataset import EuroSAT
from settings import project_root, seed
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
import torch
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

check_reproducability = False 

CONFIG = {
    "batch_size": 50,
    "num_workers": 4,
    "augmentation_mode": "weak"
}

def set_seed():
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(path, num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def get_test_predictions(model, loader: DataLoader):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())

    # Merge all batches
    all_logits = torch.cat(all_logits)
    
    eurosat_samples = loader.dataset.samples
    all_paths, all_labels = zip(*eurosat_samples)
    
    return all_logits, torch.tensor(all_labels), all_paths

def get_class_tpr(logits, labels, num_classes):
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    _, predictions = torch.max(logits, 1)
    class_total += torch.bincount(labels, minlength=num_classes)
    mask = predictions == labels
    correct_labels = labels[mask]
    class_correct += torch.bincount(correct_labels, minlength=num_classes)

    class_tpr = [float(class_correct[i]) / int(class_total[i]) for i in range(num_classes)]
    return class_tpr

def find_top_bottom_images(logits, labels, paths, classes: list, selected_classes: list, k: int = 5):
    probabilities = torch.softmax(logits, dim=1)
    top_bottom_images = {}
    
    for class_name in selected_classes:
        class_idx = classes.index(class_name)
        mask = labels == class_idx
        scores_for_class = probabilities[mask, class_idx]
        
        sorted_scores = torch.argsort(scores_for_class)
        bottom_indices = sorted_scores[:k]
        top_indices = sorted_scores[-k:]

        masked_paths = [path for path, keep in zip(paths, mask.tolist()) if keep]

        top_bottom_images[class_name]["top"] = [masked_paths[idx] for idx in top_indices]
        top_bottom_images[class_name]["bottom"] = [masked_paths[idx] for idx in bottom_indices]

    return top_bottom_images

def main():
    set_seed()
    
    models_dir = os.path.join(project_root, "models")
    output_dir = os.path.join(project_root, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = f"best_model_{CONFIG['augmentation_mode']}.pth"
    model_path = os.path.join(models_dir, model_name)
    logits_path = os.path.join(output_dir, "logits.pt")

    test_dataset = EuroSAT("EuroSAT_RGB", split="test", transform="val")
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    all_classes = test_dataset.classes
    num_classes = len(all_classes)
    model = get_model(model_path, num_classes)

    print("Running Tests...")
    logits, labels, paths = get_test_predictions(model, test_loader)
    
    if check_reproducability:
        if not os.path.exists(logits_path):
            print("No saved logits found.")
            return
        saved_logits = torch.load(logits_path)
        if torch.allclose(logits, saved_logits, atol=1e-5):
            print("Logits match!")
        else:
            print("Logits do not match!")
            
    else:
        torch.save(logits, logits_path)
        selected_classes = ["River", "Forest", "SeaLake"]
        top_bottom_images = find_top_bottom_images(logits, labels, paths, all_classes, selected_classes)

        class_tpr = get_class_tpr(logits, labels, num_classes)
        class_tpr_readable = {all_classes[i]: class_tpr[i] for i in range(len(class_tpr))}
        
        with open(os.path.join(output_dir, "test_class_tpr.json"), "w") as f:
            json.dump(class_tpr_readable, f, indent=4)

        with open(os.path.join(output_dir, "image_paths.json"), "w") as f:
            json.dump(top_bottom_images, f, indent=4)
            
        print("Results saved to /predictions.")

if __name__ == "__main__":
    main()