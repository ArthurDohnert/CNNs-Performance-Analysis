# src/utils/preprocess.py

###
### Functions that preprocess the data
###

# imports
from src.utils.datasets import TinyImageNetTrain, TinyImageNetVal
from torchvision import transforms
import torch
import os

# implementation

# reads original tiny imagenet and saves it in a single .pt file for training.
def preprocess_and_save(raw_data_dir, out_file):
    train_dir = os.path.join(raw_data_dir, "train")
    val_dir = os.path.join(raw_data_dir, "val")
    wnids_file = os.path.join(raw_data_dir, "wnids.txt")

    # transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # create datasets
    train_dataset = TinyImageNetTrain(train_dir, wnids_file, transform=train_transform)
    val_dataset = TinyImageNetVal(val_dir, wnids_file, transform=val_transform)

    # accumulate tensors in memory
    def dataset_to_tensors(dataset):
        xs, ys = [], []
        for x, y in dataset:
            xs.append(x.unsqueeze(0))  # adds batch dimension
            ys.append(torch.tensor([y]))
        return torch.cat(xs), torch.cat(ys)

    print("processing train...")
    train_x, train_y = dataset_to_tensors(train_dataset)
    print("processing validation...")
    val_x, val_y = dataset_to_tensors(val_dataset)

    # saves in a .pt
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    torch.save({
        "train": (train_x, train_y),
        "val": (val_x, val_y),
        "num_classes": len(train_dataset.class_to_idx)
    }, out_file)
    print(f"preprocessed data saved in: {out_file}")    