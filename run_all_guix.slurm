#!/bin/bash

# ==============================================================================
# SCRIPT DE TESTE 
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

CUDA_DRIVER_PATH=""

# 1. Tenta encontrar o caminho via ldconfig (método preferido)
# Ignora erros caso ldconfig não esteja disponível
CUDA_DRIVER_PATH=$(ldconfig -p 2>/dev/null | grep -om1 '.*libcuda\.so\.1$' | xargs -r dirname)

# 2. Fallback: se ldconfig falhar, usa 'find' em caminhos comuns
if [ -z "$CUDA_DRIVER_PATH" ]; then
    echo "ldconfig falhou ou não encontrou. Usando 'find' em caminhos comuns..."
    COMMON_PATHS=("/usr/lib64" "/usr/lib" "/usr/lib/x86_64-linux-gnu" "/run/opengl-driver/lib")
    
    for path in "${COMMON_PATHS[@]}"; do
        if [ -f "$path/libcuda.so.1" ]; then
            CUDA_DRIVER_PATH="$path"
            echo "Encontrado em (busca comum): $CUDA_DRIVER_PATH"
            break
        fi
    done
fi

# 3. Fallback final: busca ampla (pode ser lenta)
if [ -z "$CUDA_DRIVER_PATH" ]; then
    echo "Busca comum falhou. Usando busca ampla (pode demorar)..."
    CUDA_DRIVER_PATH=$(find / -name libcuda.so.1 -exec dirname {} \; 2>/dev/null | head -n 1)
fi

# 4. Exporta a variável se o caminho foi encontrado
if [ -n "$CUDA_DRIVER_PATH" ]; then
    echo "--- Sucesso: libcuda.so.1 encontrado em: $CUDA_DRIVER_PATH"
    # Adiciona o caminho ao LD_LIBRARY_PATH
    # O guix shell irá preservar esta variável
    export LD_LIBRARY_PATH="${CUDA_DRIVER_PATH}:${LD_LIBRARY_PATH:-}"
    echo "--- LD_LIBRARY_PATH (antes do Guix): $LD_LIBRARY_PATH"
else
    echo "!!! ERRO CRÍTICO: Não foi possível encontrar libcuda.so.1 no host."
    echo "!!! O PyTorch (torch.cuda.is_available()) irá falhar."
    exit 1 # Falha o job se não encontrar o driver
fi



# Fallbacks portáveis para área temporária/scratch
JOB_ID="${SLURM_JOB_ID:-$$}"
USER_NAME="${USER:-${LOGNAME:-user}}"

# Base preferindo $SCRATCH, depois $TMPDIR, depois /scratch/$USER, depois $HOME/.scratch
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/scratch/${USER_NAME}}}"
[ -d "${SCRATCH_BASE}" ] || SCRATCH_BASE="${HOME}/.scratch"

SCRATCH_JOB="${SCRATCH_BASE}/job_${JOB_ID}"
mkdir -p -m 700 "${SCRATCH_JOB}"

# Exporte e use só após definido
export SCRATCH="${SCRATCH_JOB}"

if [ -f "./channels.scm" ]; then
  GUIX_SHELL=(guix time-machine -C ./channels.scm -- shell -m manifest.scm \
    --preserve=^SLURM \
    --preserve=^SCRATCH$ \
    --preserve=^LD_LIBRARY_PATH$ \
    --preserve=^CUDA_VISIBLE_DEVICES$ \
    --preserve=^PYTHONUSERBASE$ \
    --preserve=^PIP_CACHE_DIR$)
else
  GUIX_SHELL=(guix shell -m manifest.scm \
    --preserve=^SLURM \
    --preserve=^SCRATCH$ \
    --preserve=^LD_LIBRARY_PATH$ \
    --preserve=^CUDA_VISIBLE_DEVICES$ \
    --preserve=^PYTHONUSERBASE$ \
    --preserve=^PIP_CACHE_DIR$)
fi

# Prefixos isolados por job dentro do SCRATCH
export PIP_CACHE_DIR="${SCRATCH}/.pip-cache"
export PYTHONUSERBASE="${SCRATCH}/.pip-user"
export PATH="${PYTHONUSERBASE}/bin:${PATH}" 


# 1 Checagem PyTorch/CUDA após instalar dependências
"${GUIX_SHELL[@]}" -- bash -c '
  set -euo pipefail

  python3 -m pip install --upgrade pip wheel setuptools

  # Instala todos do requirements EXCETO torch/vision/audio
  python3 -m pip install -r <(grep -viE "^(torch|torchvision|torchaudio)\b" requirements.txt) --user

  # PyTorch 2.2.1 + cu118 (URL sem colchetes)
  python3 -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --user

  export PYTHONPATH="$(python3 -m site --user-site):${PYTHONPATH:-}"

  python3 - <<PY
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
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma"
print(f"GPU: {gpu}")
PY
'

# 2 Staging do dataset no SCRATCH
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

# 3 Execução dentro do guix shell com user-site ativo
DATASET_PATH_TRAIN="$SCRATCH/tiny-imagenet-200/train"
DATASET_PATH_VAL="$SCRATCH/tiny-imagenet-200/val"
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

"${GUIX_SHELL[@]}" -- python3 -m src.main \
  --model_name "$1" \
  --config_path "$3" \
  --train_data_path "$DATASET_PATH_TRAIN" \
  --val_data_path "$DATASET_PATH_VAL" \
  --seed "$2"
"
echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO "
echo "========================================================="