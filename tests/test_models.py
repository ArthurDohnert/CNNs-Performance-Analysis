# tests/test_models.py

import unittest
import torch

# Importe as classes dos modelos que você quer testar
# O '..' antes de 'src' é para navegar um diretório acima a partir de 'tests/'
from src.models import mobilenet_v1 # Adicione todos os modelos aqui

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Configuração executada antes de cada teste."""
        self.batch_size = 4
        self.num_classes = 200  # Para Tiny ImageNet
        self.input_tensor = torch.randn(self.batch_size, 3, 64, 64) # Tensor de entrada mock

    def test_mobilenet_v1_forward_pass(self):
        """Testa se o MobileNetV1 processa a entrada e retorna a forma de saída correta."""
        model = mobilenet_v1.MobileNetV1(num_classes=self.num_classes)
        model.eval() # Coloca em modo de avaliação
        with torch.no_grad():
            output = model(self.input_tensor)
        
        # Verifica se a saída tem o formato (batch_size, num_classes)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    # Adicione um teste para cada um dos seus modelos aqui...
    # def test_xception_forward_pass(self):
    #     ...

if __name__ == '__main__':
    unittest.main()
