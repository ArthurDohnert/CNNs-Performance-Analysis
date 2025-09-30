# src/pipelines/train_pipeline.py

###
### Functions that manage the training workflow
###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ..utils import data_loader, logging, performance
from . import inference_pipeline # Para avaliação final

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Factory function to instantiate a model based on its name.
    Dynamically loads the model module to keep this function clean.
    """
    try:
        # Constrói o caminho para o módulo do modelo
        module_path = f"src.models.{model_name.lower()}"
        model_module = __import__(module_path, fromlist=[''])
        
        # Pega a classe do modelo (assumindo que o nome da classe é o mesmo que o nome do arquivo, capitalizado)
        model_class = getattr(model_module, model_name.upper())
        
        return model_class(num_classes=num_classes)
    except (ImportError, AttributeError):
        raise ValueError(f"Modelo '{model_name}' não encontrado ou o módulo não tem a classe correspondente.")


def run_training(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    config: dict,
    device: torch.device,
    logger: logging.Logger
) -> str:
    """
    Executes the full training and validation cycle for a model.
    
    Args:
        model_name: Name of the model architecture.
        train_data_path: Path to the training data.
        val_data_path: Path to the validation data.
        config: Dictionary with hyperparameters (learning_rate, epochs, batch_size).
        device: The device to train on (CPU or CUDA).
        logger: The logger object for tracking.
        
    Returns:
        Path to the saved trained model.
    """
    # Carrega os DataLoaders
    train_loader = data_loader.get_dataloader(train_data_path, config['batch_size'], shuffle=True)
    val_loader = data_loader.get_dataloader(val_data_path, config['batch_size'], shuffle=False)

    # Instancia o modelo, otimizador e função de perda
    model = get_model(model_name, config['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Iniciando treinamento do modelo {model_name} no dispositivo {device}.")
    perf_monitor = performance.PerformanceMonitor()
    perf_monitor.start()

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Época {epoch+1} concluída. Loss média: {avg_epoch_loss:.4f}")

        # Avaliação no final de cada época (opcional, mas recomendado)
        inference_pipeline.run_inference(model, val_loader, device, logger)

    training_perf_metrics = perf_monitor.stop()
    logger.info("Métricas de performance do treinamento:")
    logging.log_results(logger, training_perf_metrics)

    # Salva o modelo treinado
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Modelo salvo em: {model_save_path}")
    
    return model_save_path
