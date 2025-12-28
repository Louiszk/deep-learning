from settings import project_root
import matplotlib.pyplot as plt
from PIL import Image
import json
import os


def visualize_images():
    json_path = os.path.join(project_root, "predictions", "image_paths.json")
    if not os.path.exists(json_path):
        print(f"File not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    row_titles = []

    for class_name, group in data.items():
        rows.append(group["top"])
        row_titles.append(f"{class_name} (top)")

        rows.append(group["bottom"])
        row_titles.append(f"{class_name} (bottom)")

    num_rows = len(rows)
    num_cols = 5

    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 3)
    )

    for row_idx, image_paths in enumerate(rows):
        for col_idx, img_path in enumerate(image_paths):
            ax = axes[row_idx][col_idx]

            full_path = os.path.join(project_root, img_path)
            img = Image.open(full_path)

            ax.imshow(img)
            ax.axis("off")

            if col_idx == 0:
                ax.set_title(
                    row_titles[row_idx],
                    loc="left",
                    fontsize=14,
                    fontweight="semibold",
                )

    plt.tight_layout()

    output_path = json_path.replace(".json", "_grid.png")
    plt.savefig(output_path)

if __name__ == "__main__":
    visualize_images()
