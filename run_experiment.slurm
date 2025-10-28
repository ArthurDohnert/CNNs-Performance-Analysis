#!/bin/bash
# TESTE COM VENV + STAGING EM $SCRATCH (partição tupi, CUDA 11.8)

#SBATCH --job-name=pcad_sanity_test_scratch_venv
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:10:00
#SBATCH --output=logs/test_job_scratch_venv_%j.out
#SBATCH --error=logs/test_job_scratch_venv_%j.err

set -euo pipefail

# Garante diretório correto e logs
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_NAME="${1:-mobilenet_v1}"
SEED="${2:-42}"

# 1) VENV no $SCRATCH
VENV_DIR="$SCRATCH/venv_cnn_perf"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

pip install -r requirements.txt

# 3) PyTorch CUDA 11.8 para tupi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Checagem
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")
PY

# 4) Staging do dataset para $SCRATCH
rsync -avP "$HOME/datasets/tiny-imagenet-200" "$SCRATCH/"

TRAIN="$SCRATCH/tiny-imagenet-200/train"
VAL="$SCRATCH/tiny-imagenet-200/val"

# Caminho ABSOLUTO do YAML (evita 'arquivo não encontrado')
CONFIG_PATH="$SLURM_SUBMIT_DIR/configs/pcad_config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERRO: Config não encontrada: $CONFIG_PATH" >&2
  exit 1
fi

# 5) Execução principal
python -m src.main \
  --model_name "$MODEL_NAME" \
  --config_path "$CONFIG_PATH" \
  --train_data_path "$TRAIN" \
  --val_data_path "$VAL" \
  --seed "$SEED"

# 6) Finalização
deactivate || true
