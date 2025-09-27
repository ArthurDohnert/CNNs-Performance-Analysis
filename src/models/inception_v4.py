# src/models/inceptionv4.py

###
### inceptionv3 model class. the implementation was imported from timm
###

# imports
import torch
import torch.nn as nn
import timm

class InceptionV4(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(InceptionV4, self).__init__()
        # load timm model
        self.model = timm.create_model('inception_v4', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = InceptionV4(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("output shape:", y.shape)  # (4, 200)