#!/bin/bash -l


# --- Configurações do Job de Teste ---
#SBATCH --job-name=pcad_sanity_test_final
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_job_final_%j.out
#SBATCH --error=logs/test_job_final_%j.err

# --- PREPARAÇÃO DO AMBIENTE ---
echo "Inicializando o sistema de módulos..."
source /etc/profile.d/modules.sh

echo "Carregando o módulo Guix..."
module load guix

# Carrega o seu perfil de usuário por segurança
source ~/.bashrc
# ----------------------------------------------------

# --- ETAPA 1: COPIAR DADOS PARA O DISCO LOCAL (STAGING) ---
echo "========================================================="
echo "Iniciando Staging: Copiando dataset para o $SCRATCH..."
rsync -avP $HOME/datasets/tiny-imagenet-200 $SCRATCH/
echo "Staging concluído."
echo "========================================================="

# --- ETAPA 2: EXECUÇÃO USANDO DADOS LOCAIS ---
echo "Iniciando Job de Teste..."
DATASET_PATH_TRAIN="$SCRATCH/tiny-imagenet-200/train"
DATASET_PATH_VAL="$SCRATCH/tiny-imagenet-200/val"

# Argumentos recebidos do 'sbatch'
MODEL_NAME=$1
SEED=$2

guix shell -m manifest.scm -- \
     python -m src.main \
            --model_name "$MODEL_NAME" \
            --config_path "configs/pcad_test_config.yaml" \
            --train_data_path "$DATASET_PATH_TRAIN" \
            --val_data_path "$DATASET_PATH_VAL" \
            --seed "$SEED"

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO"
echo "========================================================="

