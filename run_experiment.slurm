#!/bin/bash

# --- Configurações do Slurm ---
#SBATCH --job-name=cnn_performance_test  # Nome do job
#SBATCH --partition=tupi                 # Fila de execução (partição). 
#SBATCH --nodes=1                        # Pedir 1 nó (máquina)
#SBATCH --ntasks=1                       # Pedir 1 tarefa (geralmente 1 para jobs de GPU)
#SBATCH --gres=gpu:1                     # Pedir 1 GPU
#SBATCH --time=12:00:00                  # Tempo máximo de execução (12 horas)
#SBATCH --output=logs/%x_%j_%A_%a.out    # Arquivo para salvar a saída padrão
#SBATCH --error=logs/%x_%j_%A_%a.err     # Arquivo para salvar a saída de erro

# --- Ambiente e Execução ---

# Carregar o ambiente Guix (verificar com Schnnor)
# Pode ser que `source /etc/profile` ou algo similar seja necessário primeiro.
# Se o Guix já estiver no seu PATH, esta linha pode não ser necessária.
# module load guix 

echo "Iniciando job no no: $(hostname)"
echo "Data: $(date)"
echo "Modelo a ser treinado: $1" # Recebe o nome do modelo como primeiro argumento

# Comando principal: Inicia o ambiente Guix e executa o script Python
# O -- guix shell separará os comandos do Guix dos comandos do shell interno
guix shell -m manifest.scm -- \
     python train.py --model "$1" --dataset TinyImageNet --epochs 50

echo "Job concluído com sucesso!"
echo "Data: $(date)"
