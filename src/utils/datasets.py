# src/utils/datasets.py

###
### Classes and functions for working with the dataset itself
### 

# imports
from torch.utils.data import Dataset
from PIL import Image
import os

# implementation
class TinyImageNetTrain(Dataset):
    def __init__(self, root, wnids_file, transform=None):
        self.root = root
        self.transform = transform

        # reads the 200 classes
        with open(wnids_file, "r") as f:
            self.wnids = [line.strip() for line in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        # creates image files
        self.samples = []
        for wnid in self.wnids:
            img_dir = os.path.join(root, wnid, "images")
            for img_file in os.listdir(img_dir):
                if img_file.endswith(".JPEG"):
                    self.samples.append((os.path.join(img_dir, img_file), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TinyImageNetVal(Dataset):
    def __init__(self, root, wnids_file, transform=None):
        self.root = root
        self.transform = transform

        with open(wnids_file, "r") as f:
            self.wnids = [line.strip() for line in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        val_ann_path = os.path.join(root, "val_annotations.txt")
        with open(val_ann_path, "r") as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            parts = line.strip().split("\t")
            img_file, wnid = parts[0], parts[1]
            img_path = os.path.join(root, "images", img_file)
            self.samples.append((img_path, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
