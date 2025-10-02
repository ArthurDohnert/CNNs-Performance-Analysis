#!/bin/bash

# ==============================================================================
# SCRIPT DE SUBMISSÃO SLURM PARA O PROJETO CNNS-PERFORMANCE-ANALYSIS
# ==============================================================================

# --- Configurações do Job ---
#SBATCH --job-name=cnn_experiment     # Nome padrão do job
#SBATCH --partition=tupi              # Fila de execução 
#SBATCH --nodes=1                     # Requisitar 1 nó 
#SBATCH --ntasks=1                    # Requisitar 1 tarefa
#SBATCH --gres=gpu:1                  # Requisitar 1 GPU neste nó
#SBATCH --time=01:00:00               # Tempo máximo: 12 horas
#SBATCH --output=logs/%x_%j.out       # Log de saída padrão (%x=nome, %j=ID)
#SBATCH --error=logs/%x_%j.err        # Log de saída de erro

# --- Ambiente e Execução ---
echo "========================================================="
echo "INICIANDO JOB SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nó alocado: $(hostname)"
echo "Data de início: $(date)"
echo "========================================================="

# Argumento recebido do 'sbatch': nome do modelo a ser treinado
MODEL_NAME=$1

# Verifica se o nome do modelo foi fornecido
if [ -z "$MODEL_NAME" ]; then
    echo "Erro: O nome do modelo não foi fornecido."
    exit 1
fi

# Comando principal: Inicia o ambiente Guix e executa o script principal
# Usamos 'python -m src.main' para que os imports relativos funcionem
# O script principal recebe os argumentos e orquestra os pipelines
guix shell -m manifest.scm -- \
     python -m src.main \
            --model_name "$MODEL_NAME" \
            --config_path "configs/pcad_config.yaml" \
            --train_data_path "/path/to/your/tiny-imagenet-200/train" \
            --val_data_path "/path/to/your/tiny-imagenet-200/val"

echo "========================================================="
echo "JOB CONCLUÍDO"
echo "Data de término: $(date)"
echo "========================================================="

