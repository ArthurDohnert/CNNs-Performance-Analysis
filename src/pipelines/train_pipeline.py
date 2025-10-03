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
import logging
from ..utils import data_loader, logging as custom_logging, performance
from . import inference_pipeline # Para avaliação final
from ..utils import reproducibility

def get_model(model_name: str, num_classes: int) -> torch.nn.Module:

    try:
        module_path = f"src.models.{model_name.lower()}"
        
        
        model_name_map = {
            "mobilenet_v1": "MobileNetV1",
            "vgg16": "VGG16",
            "resnet34": "ResNet34",
            "resnet101": "ResNet101",
            "xception": "Xception",
            "densenet121": "DenseNet121",
            "efficientnet_b0": "EfficientNetB0",
            "efficientnet_b7": "EfficientNetB7",
            "inception_v3": "InceptionV3",
            "inception_v4": "InceptionV4",
            "shufflenet_v2": "ShuffleNetV2",
            "squeezenet": "SqueezeNet",
            # Adicione todos os outros mapeamentos aqui
        }
        
        class_name = model_name_map.get(model_name.lower())
        if not class_name:
             raise ValueError(f"Mapeamento para o modelo '{model_name}' não encontrado.")

        # Importa o módulo dinamicamente
        model_module = __import__(module_path, fromlist=[class_name])
        
        # Pega a classe de dentro do módulo
        model_class = getattr(model_module, class_name)
        
        # Retorna a instância do modelo
        return model_class(num_classes=num_classes)
        
    except (ImportError, AttributeError, KeyError) as e:
        raise ValueError(f"Modelo '{model_name}' não encontrado ou o módulo/classe está incorreto. Erro: {e}")


def run_training(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    config: dict,
    device: torch.device,
    logger: logging.Logger,
    seed: int
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
    reproducibility.set_seed(seed)
    logger.info(f"Semente aleatória para esta execução foi fixada em: {seed}")

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
            outputs = model(inputs)
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
    custom_logging.log_results(logger, training_perf_metrics)

    # Salva o modelo treinado
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Modelo salvo em: {model_save_path}")
    
    return model_save_path
