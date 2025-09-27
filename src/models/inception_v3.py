# src/models/inceptionv3.py

###
### inceptionv3 model class. the implementation was imported from PyTorch
###

# imports
import torch
import torch.nn as nn
from torchvision.models import inception_v3

class InceptionV3(nn.Module):
    def __init__(self, num_classes=200, aux_logits=False):
        super(InceptionV3, self).__init__()
        # carregando a InceptionV3 pré-definida (sem pesos)
        self.model = inception_v3(weights=None, aux_logits=aux_logits)
        
        # ajustando primeira camada para 64x64 (stem)
        # original é 3x3 conv com stride=2 -> aqui usamos stride=1
        self.model.Conv2d_1a_3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # desativando o aux classifier se não quiser usar
        if not aux_logits:
            self.model.AuxLogits = None
        
        # ajustando classifier final
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
        

if __name__ == "__main__":
    model = InceptionV3(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("output shape:", y.shape)  # (4, 200)
