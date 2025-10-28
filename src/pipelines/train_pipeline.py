# src/pipelines/train_pipeline.py

###
### Functions that manage the training workflow
###

from sklearn import logger
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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    try:
        module_path = f"src.models.{model_name}"
        
        model_name_map = {
            "mobilenet_V1": "MobileNetV1",
            "vgg16": "VGG16",
            "resnet34": "resnet34",
            "resnet101": "resnet101",
            "xception": "Xception",
            "densenet121": "DenseNet121",
            "efficientnet_b0": "EfficientNetB0",
            "efficientnet_b7": "EfficientNetB7",
            "inception_v3": "InceptionV3",
            "inception_v4": "InceptionV4",
            "shufflenet_v2": "ShuffleNetV2",
            "squeezenet": "SqueezeNet",
        }
        
        class_name = model_name_map.get(model_name)
        if not class_name:
            raise ValueError(f"Mapeamento para o modelo '{model_name}' não encontrado.")

        model_module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(model_module, class_name)
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
    Executa o ciclo completo de treinamento e validação com AMP e LR Scheduler.
    """

    reproducibility.set_seed(seed)
    logger.info(f"Semente aleatória para esta execução foi fixada em: {seed}")
    logger.info(f"Configuração de treinamento: {config}")

    train_params = config['train_params']
    inference_params = config['inference_params']

    train_loader = data_loader.get_dataloader(
        train_data_path, 
        train_params['batch_size'], 
        shuffle=True, 
        is_train=True,  
        num_workers=16
    )
    val_loader = data_loader.get_dataloader(
        val_data_path, 
        inference_params['batch_size'], 
        shuffle=False, 
        is_train=False, 
        num_workers=16
    )

    model = get_model(model_name, config['model_params']['num_classes']).to(device)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=train_params['learning_rate'], 
        momentum=0.9, 
        weight_decay=5e-4
    )
    logger.info("Usando otimizador SGD com momento e weight decay.")
    criterion = nn.CrossEntropyLoss()

    # --- 2. CRIAÇÃO DINÂMICA DO SCHEDULER ---
    scheduler = None
    if 'scheduler_params' in config and config['scheduler_params']:
        scheduler_config = config['scheduler_params']
        scheduler_type = scheduler_config.get('type')
        
        if scheduler_type == 'StepLR':
            scheduler = StepLR(optimizer, 
                               step_size=scheduler_config['step_size'], 
                               gamma=scheduler_config['gamma'])
            logger.info(f"Usando StepLR scheduler com step_size={scheduler_config['step_size']} e gamma={scheduler_config['gamma']}.")
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, 
                                          T_max=scheduler_config['T_max'], 
                                          eta_min=scheduler_config.get('eta_min', 0))
            logger.info(f"Usando CosineAnnealingLR scheduler com T_max={scheduler_config['T_max']}.")
        # Adicione outros schedulers aqui se necessário
        else:
            logger.warning(f"Tipo de scheduler '{scheduler_type}' não reconhecido. Treinando com LR constante.")
    else:
        logger.info("Nenhum scheduler de learning rate configurado. Treinando com LR constante.")


    scaler = GradScaler(enabled=(device.type == 'cuda'))
    logger.info(f"AMP (Automatic Mixed Precision) {'ativado' if scaler.is_enabled() else 'desativado'}.")

    logger.info(f"Iniciando treinamento do modelo {model_name} no dispositivo {device}.")
    perf_monitor = performance.PerformanceMonitor()
    perf_monitor.start()

    for epoch in range(train_params['epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_params['epochs']}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Época {epoch+1} concluída. Loss média: {avg_epoch_loss:.4f}")

        # --- 3. ATUALIZAÇÃO DO SCHEDULER ---
        if scheduler:
            scheduler.step()
            # Log opcional para verificar a mudança do LR
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Taxa de aprendizado ajustada para: {current_lr:.6f}")

        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                inference_pipeline.run_inference(model, val_loader, device, logger)

    training_perf_metrics = perf_monitor.stop()
    logger.info("Métricas de performance do treinamento:")
    custom_logging.log_results(logger, training_perf_metrics)

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Modelo salvo em: {model_save_path}")
    
    return model_save_path

