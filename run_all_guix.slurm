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
#    (Assumindo que eles estão no mesmo diretório do script sbatch)
SCRIPT_DIR=$( (cd "$(dirname "$0")" && pwd) )
MANIFEST_CPU_PATH="$SCRIPT_DIR/manifest-cpu.scm"
MANIFEST_GPU_PATH="$SCRIPT_DIR/manifest-gpu.scm"
LOCAL_PACKAGE_PATH="$SCRIPT_DIR/packages" # Diretório contendo timm.scm

# 2. Lógica condicional para selecionar o ambiente
#    Isso replica a lógica original do venv, mas seleciona um manifesto.
if [ "${SLURM_JOB_PARTITION:-}" = "shared" ]; then
  echo "[INFO] Partição 'shared' detectada -> usando manifesto CPU-only."
  SELECTED_MANIFEST="$MANIFEST_CPU_PATH"
else
  echo "[INFO] Partição '${SLURM_JOB_PARTITION}' -> usando manifesto GPU (CUDA)."
  SELECTED_MANIFEST="$MANIFEST_GPU_PATH"
fi

# Validação rápida de que os arquivos existem
if [ ! -f "$SELECTED_MANIFEST" ]; then
  echo "ERRO: Arquivo de manifesto não encontrado: $SELECTED_MANIFEST"; exit 1
fi
if [ ! -d "$LOCAL_PACKAGE_PATH" ]; then
  echo "ERRO: Diretório de pacote local não encontrado: $LOCAL_PACKAGE_PATH"; exit 1
fi
echo "Manifesto selecionado: $SELECTED_MANIFEST"
echo "========================================================="

# --- ETAPA 1: COPIAR DADOS PARA O DISCO LOCAL (STAGING) ---
# (Esta seção permanece inalterada, é uma excelente prática.)
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

# O 'guix shell' substitui 'source venv/bin/activate' e 'pip install'.
#
# - --load-path: Informa ao Guix onde encontrar o módulo 'timm.scm'.[37, 41]
# - -m: Especifica o arquivo de manifesto a ser usado.[5]
# - --preserve=: Crucial! Passa as variáveis do Slurm/CUDA
#   para dentro do ambiente Guix.[2, 40]
# - --: Separa as opções do Guix do comando a ser executado.
# - bash -c "...": Executa um sub-shell com o ambiente Guix,
#   permitindo-nos replicar a lógica de 'verificação' do script original.
guix shell --load-path="$LOCAL_PACKAGE_PATH" \
           -m "$SELECTED_MANIFEST" \
           --preserve='^SLURM_' \
           --preserve='^CUDA_' \
           -- \
           bash -c "
set -euo pipefail
echo '--- Verificando Ambiente Guix ---'
# A verificação Python original agora é executada dentro do ambiente Guix
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