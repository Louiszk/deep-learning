import os

seed = 3705548

project_root = "."
datasets_root = os.path.join(project_root, "datasets")
dataset_type = "RGB"  # RGB or MS
split_size = (250, 100, 200)  # train, val, test