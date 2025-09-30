# src/models/efficientnet_b7.py

###
### EfficientNet-B7 model class. the implementation was imported from timm
###

# imports
import torch
import torch.nn as nn
import timm

class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(EfficientNetB7, self).__init__()
        # create timm model
        self.model = timm.create_model('efficientnet_b7', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = EfficientNetB7(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("output shape:", y.shape)  # (4, 200)
