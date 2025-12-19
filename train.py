from dataset import EuroSAT
from transforms import get_transforms
from settings import project_root, seed
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "num_workers": 8,
    "batch_size": 50,
    "epochs": 20,
    "learning_rate": 0.001,
    "momentum": 0.9,
    "augmentation_mode": "weak"
}

# https://docs.pytorch.org/docs/stable/notes/randomness.html
def set_seed():
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Each data loader worker should have a different (reproducable) seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model.to(device)

def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Compute gradients and update weights
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, loader, num_classes):
    model.eval()
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            class_total += torch.bincount(labels, minlength=num_classes)
            mask = predictions == labels
            correct_labels = labels[mask]
            class_correct += torch.bincount(correct_labels, minlength=num_classes)

    class_tpr = [float(class_correct[i]) / int(class_total[i]) for i in range(num_classes)]
    val_accuracy = float(class_correct.sum()) / int(class_total.sum())
    
    return val_accuracy, class_tpr

def main():
    output_dir = os.path.join(project_root, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    set_seed()
    g = torch.Generator()
    g.manual_seed(seed)
    
    print(f"Starting training.")

    train_transform = get_transforms(mode=CONFIG["augmentation_mode"])
    val_transform = get_transforms(mode="val")
    
    train_dataset = EuroSAT("EuroSAT_RGB", "train", transform=train_transform)
    val_dataset = EuroSAT("EuroSAT_RGB", "val", transform=val_transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"]
    )
    
    model = get_model(num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"])
    
    training_results = {
        "config": CONFIG,
        "val_accuracy": [],
        "class_tpr": {class_name: [] for class_name in class_names}
    }

    best_accuracy = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_epoch(model, train_loader, loss_fn, optimizer)
        val_accuracy, class_tpr = validate(model, val_loader, num_classes)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_name = f"best_model_{CONFIG['augmentation_mode']}.pth"
            torch.save(model.state_dict(), os.path.join(output_dir, save_name))
        
        training_results["val_accuracy"].append(val_accuracy)
        for i, name in enumerate(class_names):
            training_results["class_tpr"][name].append(class_tpr[i])

    log_filename = f"training_results_{CONFIG['augmentation_mode']}.json"
    with open(os.path.join(output_dir, log_filename), "w") as f:
        json.dump(training_results, f)

    print("\nTraining complete.\nResults saved to /models.")

if __name__ == "__main__":
    main()