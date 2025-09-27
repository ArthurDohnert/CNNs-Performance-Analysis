# src/models/xception.py

###
### Xception model class. the implementation was imported from timm
###

# imports
import torch
import torch.nn as nn
import timm

class Xception(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(Xception, self).__init__()
        # load timm model
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Xception(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("output shape:", y.shape)  # (4, 200)