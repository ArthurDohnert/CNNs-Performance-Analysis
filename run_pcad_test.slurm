#!/bin/bash

# ==============================================================================
# SCRIPT DE TESTE DE SANIDADE NO PCAD
# ==============================================================================

# --- Configurações do Job de Teste ---
#SBATCH --job-name=pcad_sanity_test      
#SBATCH --partition=shared               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                    
#SBATCH --time=00:10:00                  
#SBATCH --output=logs/test_job_%j.out    
#SBATCH --error=logs/test_job_%j.err

# --- Ambiente e Execução ---
echo "========================================================="
echo "INICIANDO JOB DE TESTE NO PCAD"
echo "Job ID: $SLURM_JOB_ID | Job Name: $SLURM_JOB_NAME"
echo "Nó alocado: $(hostname) | Data: $(date)"
echo "========================================================="

# Argumentos recebidos do 'sbatch'
MODEL_NAME=$1
SEED=$2

DATASET_PATH_TRAIN="/home/user/ehdmenezes/datasets/tiny-imagenet-200/train"
DATASET_PATH_VAL="/home/user/ehdmenezes/datasets/tiny-imagenet-200/val"

# Comando principal
# Executa o main.py usando o Guix, com o modelo e a seed recebidos como argumento
guix shell -m manifest.scm -- \
     python -m src.main \
            --model_name "$MODEL_NAME" \
            --config_path "configs/pcad_test_config.yaml" \
            --train_data_path "$DATASET_PATH_TRAIN" \
            --val_data_path "$DATASET_PATH_VAL" \
            --seed "$SEED"

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO | Data: $(date)"
echo "========================================================="
