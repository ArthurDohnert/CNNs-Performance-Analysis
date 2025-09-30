# tests/test_utils.py

import unittest
import time

# Importe os módulos de utilitários
from ..src.utils import metrics, performance

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 1, 2, 0, 1, 2]
        self.y_pred = [0, 2, 1, 0, 0, 1]

    def test_accuracy(self):
        """Testa o cálculo da acurácia."""
        # 3 predições corretas de 6
        expected_accuracy = 3 / 6
        self.assertAlmostEqual(metrics.calculate_accuracy(self.y_true, self.y_pred), expected_accuracy)

    def test_precision_recall_f1(self):
        """Testa se as métricas do sklearn são calculadas sem erros."""
        # Não testamos o valor exato, mas garantimos que retornam um float entre 0 e 1
        precision = metrics.calculate_precision(self.y_true, self.y_pred)
        recall = metrics.calculate_recall(self.y_true, self.y_pred)
        f1 = metrics.calculate_f1_score(self.y_true, self.y_pred)
        
        self.assertIsInstance(precision, float)
        self.assertTrue(0 <= precision <= 1)
        self.assertIsInstance(recall, float)
        self.assertTrue(0 <= recall <= 1)
        self.assertIsInstance(f1, float)
        self.assertTrue(0 <= f1 <= 1)


class TestPerformanceMonitor(unittest.TestCase):

    def test_monitor_workflow(self):
        """Testa o fluxo de start/stop do PerformanceMonitor."""
        monitor = performance.PerformanceMonitor()
        
        monitor.start()
        time.sleep(0.1) # Simula um trabalho computacional
        results = monitor.stop()
        
        # Verifica se o dicionário de resultados contém as chaves esperadas
        self.assertIn("elapsed_time_seconds", results)
        self.assertIn("cpu_usage_percent", results)
        self.assertIn("ram_usage_bytes", results)
        
        # Verifica se o tempo decorrido é positivo
        self.assertGreater(results["elapsed_time_seconds"], 0.0)

if __name__ == '__main__':
    unittest.main()
