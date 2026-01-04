from settings import project_root
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os

def plot_training_results(training_type):
    results_path = os.path.join(project_root, "models", f"training_results_{training_type}.json")
    if not os.path.exists(results_path):
        print(f"Skipping {results_path}.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    val_accuracy: list = data["val_accuracy"]
    class_tprs = data["class_tpr"]

    epochs = list(range(1, len(val_accuracy) + 1))

    best_val = max(val_accuracy)
    best_epoch = val_accuracy.index(best_val) + 1

    print(f"[{training_type}]\nBest validation accuracy: {best_val} (Epoch {best_epoch})")

    fig, ax = plt.subplots(figsize=(12, 8))

    for class_name, tpr_values in class_tprs.items():
        ax.plot(
            epochs,
            tpr_values,
            label=class_name,
            linewidth=1.5,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("True Positive Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Class TPR ({training_type})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    fig.tight_layout()

    plot_path = results_path.replace(".json", "_class_tpr.png")
    fig.savefig(plot_path)

if __name__ == "__main__":
    trainings = ["rgb_weak", "rgb_strong", "ms"]

    for training_type in trainings:
        plot_training_results(training_type)
