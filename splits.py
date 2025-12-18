from settings import project_root, datasets_root
from sklearn.model_selection import train_test_split
import json
import os

seed = 3705548

def get_filenames(dataset_path: str):
    filenames_per_class = {}
    class_names = sorted(os.listdir(dataset_path))
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        class_file_paths = os.listdir(class_path)
        image_ext = [".jpg", ".tif"]
        filenames_per_class[class_name] = [
            os.path.join(class_name, cfp) for cfp in class_file_paths if os.path.splitext(cfp)[1] in image_ext
            ]
    
    return filenames_per_class


def split_dataset(filenames_per_class: dict, split_size: tuple):
    train_files = {}
    val_files = {}
    test_files = {}
    train_size, val_size, test_size = split_size

    for class_name, filenames in filenames_per_class.items():
        full_train, test = train_test_split(filenames, test_size=test_size, train_size=train_size + val_size, random_state=seed)
        train, val = train_test_split(full_train, test_size=val_size, train_size=train_size, random_state=seed)
        train_files[class_name], val_files[class_name], test_files[class_name] = train, val, test
    
    return train_files, val_files, test_files


def verify_splits(train: dict, val: dict, test: dict):
    for class_name in train.keys():
        combined_files = train[class_name] + val[class_name] + test[class_name]
        if len(combined_files) > len(set(combined_files)):
            raise ValueError("Splits contain duplicated entries.")

    print("Verification passed; splits are disjoint.")


def save_splits(train: dict, val: dict, test: dict, dataset_splits_path: str):
    combined_splits = {"train": train, "val": val, "test": test}

    with open(dataset_splits_path, "w") as f:
        json.dump(combined_splits, f)
    
    return combined_splits

def create_splits(dataset_type: str, split_size: tuple):
    dataset_path = os.path.join(datasets_root, dataset_type)
    splits_path = os.path.join(project_root, "splits")
    os.makedirs(splits_path, exist_ok=True)
    dataset_splits_path = os.path.join(splits_path, f"{dataset_type}_splits.json")

    if os.path.exists(dataset_splits_path):
        with open(dataset_splits_path, "r") as f:
            dataset_splits = json.load(f)
        if all([
            dataset_splits.get("train") and all(len(split) == split_size[0] for split in dataset_splits["train"].values()),
            dataset_splits.get("val") and all(len(split) == split_size[1] for split in dataset_splits["val"].values()),
            dataset_splits.get("test") and all(len(split) == split_size[2] for split in dataset_splits["test"].values())
            ]):
            return dataset_splits
    
    all_filenames_per_class = get_filenames(dataset_path)
    train_files, val_files, test_files = split_dataset(all_filenames_per_class, split_size)
    
    verify_splits(train_files, val_files, test_files)
    dataset_splits = save_splits(train_files, val_files, test_files, dataset_splits_path)
    
    print(f"Split complete.")
    return dataset_splits


if __name__ == "__main__":
    create_splits("EuroSAT_RGB", (400, 200, 400))