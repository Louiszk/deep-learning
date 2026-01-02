# Deep-Learning EuroSAT

## Setup

### Environment

```bash
python -m venv deepvenv
source deepvenv/bin/activate
pip install -r requirements.txt
```

### Dataset

Paper: EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
Download: https://zenodo.org/records/7711810#.ZAm3k-zMKEA

```
unzip EuroSAT_RGB.zip -d datasets/
unzip EuroSAT_MS.zip -d datasets/
```

## Settings

```python
# settings.py
project_root = "/path/to/your/project_root"  # Absolute path to this folder
datasets_root = "/path/to/your/datasets"  # Folder containing EuroSAT_RGB/ and EuroSAT_MS/
dataset_type = "RGB"  # Switch between "RGB" and "MS"
split_size = (250, 100, 200)  # Split size for train, val, test for each of the 10 classes
```

## Execution

###  RGB Classification

Ensure dataset_type = "RGB" in `settings.py`
Update the CONFIG in `train.py`; choose between "weak" or "strong" augmentation.

```bash
python train.py
python predict.py
```

### Multispectral Classification (Late Fusion)

Ensure dataset_type = "MS" in `settings.py`
Update the CONFIG in `train.py`; augmentation is forced to "weak" for MS data

```bash
python train.py
python predict.py
```

## Visualization

Generates performance graphs and top/bottom-5 image grids.

```bash
python visualize_training.py
python visualize_images.py

```