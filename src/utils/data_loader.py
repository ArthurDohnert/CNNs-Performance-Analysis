# src/utils/data_loader.py

###
### Functions that loads the data
###

# imports
import torch
from torch.utils.data import TensorDataset, DataLoader

# implementations
def load_preprocessed(path, batch_size=64):
    data = torch.load(path)
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    num_classes = data["num_classes"]

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes