# src/models/densenet121.py

###
### densenet121 model class. the implementation was imported from PyTorch
###

#imports
import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNet121(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(DenseNet121, self).__init__()
        # carregando DenseNet121 (sem pesos por default)
        self.model = densenet121(weights=None if not pretrained else 'DEFAULT')
        
        # ajustar primeira camada se necessário (opcional, normalmente 7x7 stride=2 -> para 64x64 podemos usar stride=1)
        self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.pool0 = nn.Identity()  # remover maxpool inicial para não reduzir demais a resolução
        
        # ajustar o classifier final
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = DenseNet121(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    print("output shape:", y.shape)  # (4, 200)
