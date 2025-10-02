#!/bin/bash

# ==============================================================================
# SCRIPT DE TESTE DE SANIDADE NO PCAD
# ==============================================================================

# --- Configurações do Job de Teste ---
#SBATCH --job-name=pcad_sanity_test      # Nome do job de teste
#SBATCH --partition=shared               # << IMPORTANTE: Usa a fila compartilhada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                     # Pede 1 GPU (se disponível na fila shared)
#SBATCH --time=00:15:00                  # << IMPORTANTE: Tempo curto, 15 minutos
#SBATCH --output=logs/test_job_%j.out    # Arquivo de log específico para o teste
#SBATCH --error=logs/test_job_%j.err

# --- Ambiente e Execução ---
echo "========================================================="
echo "INICIANDO JOB DE TESTE NO PCAD"
echo "Job ID: $SLURM_JOB_ID"
echo "Nó alocado: $(hostname)"
echo "Data de início: $(date)"
echo "========================================================="

# **AÇÃO NECESSÁRIA**: Ajuste os caminhos abaixo para os seus diretórios no PCAD
DATASET_PATH_TRAIN="/home/user/ehdmenezes/datasets/tiny-imagenet-200/train"
DATASET_PATH_VAL="/home/user/ehdmenezes/datasets/tiny-imagenet-200/val"

# Comando principal
# Executa o main.py usando o Guix, com o modelo MobileNetV1 e o config de teste
guix shell -m manifest.scm -- \
     python -m src.main \
            --model_name "mobilenet_v1" \
            --config_path "configs/pcad_test_config.yaml" \
            --train_data_path "$DATASET_PATH_TRAIN" \
            --val_data_path "$DATASET_PATH_VAL"

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO"
echo "Data de término: $(date)"
echo "========================================================="
