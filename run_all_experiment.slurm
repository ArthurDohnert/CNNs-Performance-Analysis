#!/bin/bash

# ==============================================================================
# SCRIPT DE TESTE OTIMIZADO PARA I/O USANDO $SCRATCH (VERSÃO COM VENV)
# ==============================================================================

# --- Configurações do Job de Teste ---
#SBATCH --job-name=experiments_pcad_venv
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_job_scratch_venv_%j.out
#SBATCH --error=logs/test_job_scratch_venv_%j.err

set -euo pipefail

# --- PREPARAÇÃO DO AMBIENTE PYTHON (VENV) ---

#  Cria uma venv dentro do $SCRATCH (rápido e isolado por job)
VENV_DIR="$SCRATCH/venv_cnn_perf"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

#  Atualiza pip e instala dependências do projeto
python -m pip install --upgrade pip wheel setuptools

pip install -r requirements.txt
if [[ "${SLURM_JOB_PARTITION}" == "shared" ]]; then
  echo "[INFO] Partição shared detectada -> instalando PyTorch CPU-only e desabilitando CUDA"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  export CUDA_VISIBLE_DEVICES=""        # garante CPU-only
else
  echo "[INFO] Partição ${SLURM_JOB_PARTITION} -> instalando PyTorch cu118"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

#  Verifica CUDA
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU detectada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")
PY

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

python -m src.main \
       --model_name "$1" \
       --config_path "configs/pcad_config.yaml" \
       --train_data_path "$DATASET_PATH_TRAIN" \
       --val_data_path "$DATASET_PATH_VAL" \
       --seed "$2"

# --- ETAPA 3: LIMPEZA OPCIONAL ---
# Desativa a venv (opcional)
deactivate || true

# Resultados (logs e modelos) permanecem no diretório de submissão (HOME)

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO (VENV)"
echo "========================================================="

