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

if [ -f "./channels.scm" ]; then
  GUIX_SHELL=(guix time-machine -C ./channels.scm -- shell -m manifest.scm --pure --preserve=^SLURM)
else
  GUIX_SHELL=(guix shell -m manifest.scm --pure --preserve=^SLURM)
fi

export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
export PYTHONUSERBASE="$SCRATCH/.pip-user"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUSERBASE="$SCRATCH/.pip-user" 


# 1) Checagem PyTorch/CUDA após instalar dependências
"${GUIX_SHELL[@]}" -- bash -c '
  set -euo pipefail
  export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
  export PYTHONUSERBASE="$SCRATCH/.pip-user"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONUSERBASE="$SCRATCH/.pip-user"   

  python -m pip install --upgrade pip wheel setuptools

  # Instala todos do requirements EXCETO torch/vision/audio para depois fixar cu118
  pip install -r <(grep -viE "^(torch|torchvision|torchaudio)\b" requirements.txt) --user

  # PyTorch 2.2.1 + cu118 (wheels oficiais com runtime CUDA)
  pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --user

  # Garante que o Python enxergue o user-site
  export PYTHONPATH="$(python -m site --user-site):${PYTHONPATH:-}"

 python - <<PY
 import torch, ctypes, os
 print("torch:", torch.version)
 print("torch.version.cuda:", torch.version.cuda)
 print("built_with_cuda:", torch.backends.cuda.is_built())
 print("is_available:", torch.cuda.is_available())
 print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")) 
 try:
 ctypes.CDLL("libcuda.so.1"); print("libcuda: OK")
 except OSError as e:
 print("libcuda: FAIL ->", e)
 print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"}")
PY
'

# 2) Staging do dataset no SCRATCH
echo "========================================================="
echo "Iniciando Staging: Copiando dataset para o \$SCRATCH..."
if [ -d "$HOME/datasets/tiny-imagenet-200/val/images" ]; then
  echo "ERRO: Origem tem val/images (formato original); reorganize no HOME antes."; exit 11
fi
n_val_home=$(find "$HOME/datasets/tiny-imagenet-200/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n_val_home" -eq 200 ] || { echo "ERRO: Origem val tem $n_val_home dirs (esperado=200)."; exit 12; }

rm -rf "$SCRATCH/tiny-imagenet-200"
rsync -a --delete "$HOME/datasets/tiny-imagenet-200/" "$SCRATCH/tiny-imagenet-200/"

if [ -d "$SCRATCH/tiny-imagenet-200/val/images" ]; then
  echo "ERRO: val/images encontrado no SCRATCH (inválido p/ ImageFolder)."; exit 21
fi
n_val_scratch=$(find "$SCRATCH/tiny-imagenet-200/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n_val_scratch" -eq 200 ] || { echo "ERRO: SCRATCH val tem $n_val_scratch dirs (esperado=200)."; exit 22; }
echo "Staging concluído e validado."
echo "========================================================="

# 3) Execução dentro do guix shell com user-site ativo
DATASET_PATH_TRAIN="$SCRATCH/tiny-imagenet-200/train"
DATASET_PATH_VAL="$SCRATCH/tiny-imagenet-200/val"
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

"${GUIX_SHELL[@]}" -- bash -c "
  set -euo pipefail
  export PYTHONUSERBASE='$PYTHONUSERBASE'
  export PYTHONPATH=\"\$(python -m site --user-site):\${PYTHONPATH:-}\"
  python -m src.main \
    --model_name \"$1\" \
    --config_path \"$3\" \
    --train_data_path \"$DATASET_PATH_TRAIN\" \
    --val_data_path \"$DATASET_PATH_VAL\" \
    --seed \"$2\"
"
echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO (GUIX GPU‑only, sem venv)"
echo "========================================================="
