# tests/test_pipelines.py

import unittest
from unittest.mock import patch, MagicMock
import torch

from ..src.pipelines import train_pipeline, inference_pipeline, analysis_pipeline

class TestPipelines(unittest.TestCase):

    def test_train_pipeline_get_model(self):
        """Testa a factory function 'get_model' no pipeline de treino."""
        # Testa se consegue instanciar um modelo conhecido
        model = train_pipeline.get_model('VGG16', num_classes=10)
        self.assertIsInstance(model, torch.nn.Module)
        
        # Testa se levanta um erro para um modelo desconhecido
        with self.assertRaises(ValueError):
            train_pipeline.get_model('ModeloInexistente', num_classes=10)

    @patch('src.pipelines.train_pipeline.data_loader')
    @patch('src.pipelines.train_pipeline.torch.save')
    def test_train_pipeline_smoke_test(self, mock_torch_save, mock_data_loader):
        """
        Teste de fumaça: executa o pipeline de treino com dados falsos 
        para garantir que ele roda sem erros.
        """
        # Configura os mocks
        mock_data_loader.get_dataloader.return_value = [(torch.randn(2, 3, 64, 64), torch.randint(0, 10, (2,)))]
        mock_logger = MagicMock()

        # Executa o pipeline
        try:
            train_pipeline.run_training(
                model_name='VGG16',
                train_data_path='/fake/path',
                val_data_path='/fake/path',
                config={'epochs': 1, 'batch_size': 2, 'learning_rate': 0.001, 'num_classes': 10},
                device=torch.device('cpu'),
                logger=mock_logger
            )
        except Exception as e:
            self.fail(f"run_training() levantou uma exceção inesperada: {e}")

        # Verifica se a função para salvar o modelo foi chamada
        mock_torch_save.assert_called_once()

    @patch('src.pipelines.inference_pipeline.metrics')
    def test_inference_pipeline_smoke_test(self, mock_metrics):
        """Teste de fumaça para o pipeline de inferência."""
        mock_model = torch.nn.Linear(10, 2)
        mock_dataloader = [(torch.randn(4, 10), torch.randint(0, 2, (4,)))]
        mock_logger = MagicMock()

        try:
            inference_pipeline.run_inference(mock_model, mock_dataloader, torch.device('cpu'), mock_logger)
        except Exception as e:
            self.fail(f"run_inference() levantou uma exceção inesperada: {e}")
        
        # Verifica se as funções de métrica foram chamadas
        mock_metrics.calculate_accuracy.assert_called_once()
        mock_metrics.calculate_precision.assert_called_once()

if __name__ == '__main__':
    unittest.main()
