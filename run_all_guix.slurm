#!/bin/bash

# ==============================================================================
# SCRIPT DE TESTE OTIMIZADO PARA I/O E REPRODUTIBILIDADE (VERSÃO COM GUIX)
# ==============================================================================

# --- Configurações do Job de Teste ---
#SBATCH --job-name=experiments_pcad_guix
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_job_scratch_guix_%j.out
#SBATCH --error=logs/test_job_scratch_guix_%j.err

set -euo pipefail

# --- PREPARAÇÃO DO AMBIENTE GUIX (DECLARATIVO) ---
echo "========================================================="
echo "Preparando ambiente Guix..."

# 1. Defina os caminhos para seus manifestos e pacotes locais
PROJECT_DIR="$HOME/CNNs-Performance-Analysis"
MANIFEST_PATH="$PROJECT_DIR/manifest.scm"

# Validação rápida de que os arquivos existem
if [ ! -f "$MANIFEST_PATH" ]; then
  echo "ERRO: Arquivo de manifesto não encontrado: $MANIFEST_PATH "; exit 1
fi

echo "Manifesto selecionado: $MANIFEST_PATH"
echo "========================================================="

# --- ETAPA 1: COPIAR DADOS PARA O DISCO LOCAL (STAGING) ---
echo "========================================================="
echo "Iniciando Staging: Copiando dataset para o $SCRATCH..."

# 1) Sanidade no origem (HOME)
if [ -d "$HOME/datasets/tiny-imagenet-200/val/images" ]; then
  echo "ERRO: Origem tem val/images (val está no formato original); reorganize no HOME antes."; exit 11
fi
n_val_home=$(find "$HOME/datasets/tiny-imagenet-200/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$n_val_home" -ne 200 ]; then
  echo "ERRO: Origem val tem $n_val_home dirs (esperado=200)."; exit 12
fi


# 2) Cópia limpa para o SCRATCH
rm -rf "$SCRATCH/tiny-imagenet-200"
rsync -a --delete "$HOME/datasets/tiny-imagenet-200/" "$SCRATCH/tiny-imagenet-200/"

# 3) Sanidade no destino (SCRATCH)
if [ -d "$SCRATCH/tiny-imagenet-200/val/images" ]; then
  echo "ERRO: val/images encontrado no SCRATCH (dataset inválido para ImageFolder)."; exit 21
fi

n_val_scratch=$(find "$SCRATCH/tiny-imagenet-200/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$n_val_scratch" -ne 200 ]; then
  echo "ERRO: SCRATCH val tem $n_val_scratch dirs (esperado=200)."; exit 22
fi

echo "Staging concluído e validado."
echo "========================================================="

# --- ETAPA 2: EXECUÇÃO USANDO DADOS LOCAIS E AMBIENTE GUIX ---
echo "Iniciando Job de Teste com Guix..."

DATASET_PATH_TRAIN="$SCRATCH/tiny-imagenet-200/train"
DATASET_PATH_VAL="$SCRATCH/tiny-imagenet-200/val"

export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

guix shell -m "$MANIFEST_PATH" --pure \
  --preserve=^PATH \
  --preserve=^LD_LIBRARY_PATH \
  --preserve=^CUDA_HOME \
  --preserve=^SLURM_ \
  -- bash -c "
set -euo pipefail
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm pynvml huggingface-hub
echo '--- Verificando Ambiente Guix + Pip ---'
python - <<'PY'
import torch
import os
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
  print(f'GPU detectada: {torch.cuda.get_device_name(0)}')
else:
  print('GPU detectada: Nenhuma')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")}')
print(f'CUDA_HOME (dentro do Guix): {os.environ.get("CUDA_HOME", "N/A")}')
PY
echo '---------------------------------'

echo 'Iniciando src.main...'
python -m src.main \
       --model_name \"$1\" \
       --config_path \"$3\" \
       --train_data_path \"$DATASET_PATH_TRAIN\" \
       --val_data_path \"$DATASET_PATH_VAL\" \
       --seed \"$2\"
"

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO (GUIX)"
echo "========================================================="