#!/bin/bash

# ==============================================================================
# SCRIPT DE SUBMISSÃO SLURM GENÉRICO (TEMPO DEFINIDO NA CHAMADA)
# ==============================================================================

# --- Configurações do Job (sem o tempo) ---
#SBATCH --job-name=cnn_experiment     # Nome padrão, será sobrescrito
#SBATCH --partition=tupi              # Fila para GPUs RTX 4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Ambiente e Execução ---
echo "========================================================="
echo "INICIANDO JOB SLURM"
echo "Job ID: $SLURM_JOB_ID | Job Name: $SLURM_JOB_NAME"
echo "Nó alocado: $(hostname) | Data: $(date)"
echo "Tempo requisitado: $SLURM_JOB_TIME_LIMIT"
echo "========================================================="

# Argumentos recebidos do 'sbatch'
MODEL_NAME=$1
SEED=$2 # Assumindo que você está usando a versão com semente

# **AÇÃO NECESSÁRIA**: Ajuste os caminhos abaixo para os seus diretórios no PCAD
DATASET_PATH_TRAIN="/home/user/ehdmenezes/datasets/tiny-imagenet-200/train"
DATASET_PATH_VAL="/home/user/ehdmenezes/datasets/tiny-imagenet-200/val"

# Comando principal
guix shell -m manifest.scm -- \
     python -m src.main \
            --model_name "$MODEL_NAME" \
            --config_path "configs/pcad_config.yaml" \
            --train_data_path "$DATASET_PATH_TRAIN" \
            --val_data_path "$DATASET_PATH_VAL" \
            --seed "$SEED" # Não esqueça de passar a semente

echo "========================================================="
echo "JOB CONCLUÍDO | Data: $(date)"
echo "========================================================="
