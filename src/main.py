# src/main.py

"""
Ponto de entrada principal para orquestrar os pipelines de treinamento e avaliação.
"""

import argparse
import os
import torch
import yaml
from datetime import datetime

# Importa os pipelines e utilitários do seu projeto
from .pipelines import train_pipeline, inference_pipeline
from .utils import logging, data_loader, performance

def main():
    parser = argparse.ArgumentParser(description="Ponto de Entrada Principal para Análise de Performance de CNNs")
    
    # Argumentos principais
    parser.add_argument('--model_name', type=str, required=True, help='Nome da arquitetura do modelo (ex: VGG16).')
    parser.add_argument('--config_path', type=str, required=True, help='Caminho para o arquivo de configuração (ex: configs/base_config.yaml).')
    parser.add_argument('--train_data_path', type=str, required=True, help='Caminho para o diretório de dados de treino.')
    parser.add_argument('--val_data_path', type=str, required=True, help='Caminho para o diretório de dados de validação/teste.')

    args = parser.parse_args()

    # --- 1. Carregar Configurações ---
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração não encontrado em '{args.config_path}'")
        return
        
    # --- 2. Configurar Logger ---
    experiment_name = f"{args.model_name}_epochs_{config['train_params']['epochs']}_bs_{config['train_params']['batch_size']}"
    log_file_path = f"logs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.setup_logger(log_name=experiment_name, log_file=log_file_path)

    # Loga a configuração completa antes de começar
    config_to_log = {
        "Model Name": args.model_name,
        "Config File": args.config_path,
        "Train Data": args.train_data_path,
        "Validation Data": args.val_data_path,
        **config
    }
    logging.log_experiment_config(logger, config_to_log)
    
    # --- 3. Orquestrar os Pipelines ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # O bloco try...finally garante que o shutdown do monitor de performance seja chamado
    try:
        # --- TREINAMENTO ---
        # Combina os parâmetros de treino e do modelo para o pipeline
        training_config = {
            **config['train_params'], 
            'num_classes': config['model_params']['num_classes']
        }

        trained_model_path = train_pipeline.run_training(
            model_name=args.model_name,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            config=training_config,
            device=device,
            logger=logger
        )
        logger.info(f"Pipeline de treinamento concluído. Modelo salvo em: {trained_model_path}")

        # --- INFERÊNCIA (AVALIAÇÃO FINAL) ---
        logger.info("Iniciando pipeline de inferência para avaliação final no conjunto de validação...")
        
        val_loader = data_loader.get_dataloader(
            args.val_data_path, 
            config['inference_params']['batch_size'], 
            shuffle=False
        )
        
        model = train_pipeline.get_model(args.model_name, config['model_params']['num_classes'])
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        
        inference_pipeline.run_inference(model, val_loader, device, logger)
        
        logger.info("Projeto executado com sucesso!")

    except Exception as e:
        logger.error(f"Ocorreu um erro fatal durante a execução do projeto: {e}", exc_info=True)


if __name__ == '__main__':
    main()
