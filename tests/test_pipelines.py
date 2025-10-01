# tests/test_pipelines.py

import unittest
import torch
from unittest.mock import patch, MagicMock

from src.pipelines import train_pipeline, inference_pipeline

class TestPipelines(unittest.TestCase):

    def test_train_pipeline_get_model(self):
        """Testa a factory function 'get_model' no pipeline de treino."""
        model = train_pipeline.get_model('mobilenet_v1', num_classes=10)
        self.assertIsInstance(model, torch.nn.Module)
        
        with self.assertRaises(ValueError):
            train_pipeline.get_model('ModeloInexistente', num_classes=10)

    @patch('src.pipelines.train_pipeline.data_loader')
    @patch('src.pipelines.train_pipeline.torch.save')
    @patch('src.pipelines.train_pipeline.inference_pipeline.run_inference') # Mocka a avaliação
    def test_train_pipeline_smoke_test(self, mock_run_inference, mock_torch_save, mock_data_loader):
        """
        Teste de fumaça: executa o pipeline de treino com dados falsos 
        para garantir que ele roda sem erros.
        """
        mock_data_loader.get_dataloader.return_value = [(torch.randn(2, 3, 64, 64), torch.randint(0, 10, (2,)))]
        mock_logger = MagicMock()

        try:
            train_pipeline.run_training(
                model_name='mobilenet_v1',
                train_data_path='/fake/path',
                val_data_path='/fake/path',
                config={'epochs': 1, 'batch_size': 2, 'learning_rate': 0.001, 'num_classes': 10},
                device=torch.device('cpu'),
                logger=mock_logger
            )
        except Exception as e:
            self.fail(f"run_training() levantou uma exceção inesperada: {e}")

        mock_torch_save.assert_called_once()
        mock_run_inference.assert_called_once() # Verifica se a avaliação foi chamada

    # --- TESTE CORRIGIDO ---
    @patch('src.pipelines.inference_pipeline.custom_logging.log_results') # Mocka a função específica de log
    @patch('src.pipelines.inference_pipeline.metrics')
    def test_inference_pipeline_smoke_test(self, mock_metrics, mock_log_results):
        """Teste de fumaça para o pipeline de inferência."""
        mock_model = torch.nn.Linear(10, 2)
        mock_dataloader = [(torch.randn(4, 10), torch.randint(0, 2, (4,)))]
        mock_logger = MagicMock() # O logger ainda é um mock, mas não será usado para formatação

        try:
            inference_pipeline.run_inference(mock_model, mock_dataloader, torch.device('cpu'), mock_logger)
        except Exception as e:
            self.fail(f"run_inference() levantou uma exceção inesperada: {e}")
        
        # Verifica se as funções de métrica e log foram chamadas
        mock_metrics.calculate_accuracy.assert_called_once()
        mock_log_results.assert_called_once()


if __name__ == '__main__':
    unittest.main()
