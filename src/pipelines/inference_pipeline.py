# src/pipelines/inference_pipeline.py

###
### Functions that manage the inference workflow
###

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils import metrics, performance, logging

def run_inference(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    logger: logging.Logger
) -> dict:
    """
    Runs inference on a given model and dataloader, and computes performance metrics.
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    
    logger.info(f"Iniciando inferência com o modelo {model.__class__.__name__}...")
    perf_monitor = performance.PerformanceMonitor()
    perf_monitor.start()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inferindo"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    perf_metrics = perf_monitor.stop()
    
    quality_metrics = {
        "accuracy": metrics.calculate_accuracy(all_labels, all_predictions),
        "precision": metrics.calculate_precision(all_labels, all_predictions),
        "recall": metrics.calculate_recall(all_labels, all_predictions),
        "f1_score": metrics.calculate_f1_score(all_labels, all_predictions),
    }

    final_metrics = {**perf_metrics, **quality_metrics}
    logging.log_results(logger, final_metrics)
    
    return final_metrics
