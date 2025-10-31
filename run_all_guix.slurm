#!/bin/bash
# ==============================================================================
# SCRIPT DE TESTE OTIMIZADO PARA I/O USANDO $SCRATCH (VERSÃO COM GUIX + GPU)
# ==============================================================================

# --- Configurações do Job ---
#SBATCH --job-name=experiments_pcad_guix
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_job_scratch_guix_%j.out
#SBATCH --error=logs/test_job_scratch_guix_%j.err

set -euo pipefail

# --- Parâmetros obrigatórios ---
# $1 = model_name, $2 = seed, $3 = config_path
if [[ $# -lt 3 ]]; then
  echo "Uso: sbatch este_script.sh <model_name> <seed> <config_path>"
  exit 2
fi

# --- Caminhos do dataset (HOME -> SCRATCH) ---
DATASET_SRC_HOME="$HOME/datasets/tiny-imagenet-200"
DATASET_DST_SCRATCH="$SCRATCH/tiny-imagenet-200"

# --- Manifest do Guix (PyTorch com CUDA) ---
# Observação: requer canais não‑livres que disponibilizam CUDA e PyTorch com CUDA (ex.: guix-science-nonfree / guix-hpc-non-free / nonguix). [web:13][web:1][web:4]
# Salva o manifest no diretório do job para reprodutibilidade.
mkdir -p guix
MANIFEST_FILE="guix/manifest-cnn-perf-gpu.scm"
cat > "$MANIFEST_FILE" <<'SCM'
(use-modules (guix profiles))
;; Importante:
;; - O pacote "python-pytorch" oficial do Guix é CPU-only; usar uma variante com CUDA via canais não-livres, tipicamente chamada "python-pytorch-with-cudaXX". [web:3][web:17]
;; - Inclua CUDA Toolkit e cuDNN dos canais não-livres (nomes variam por canal, ex.: "cuda-toolkit", "cuda", "cudnn"). [web:13][web:18]
(specifications->manifest
  '(
    "python"                       ; interpretador
    "python-pip"                   ; opcional para libs auxiliares
    "python-pytorch-with-cuda11"   ; variante CUDA (ajuste conforme canal)
    "cuda-toolkit"                 ; CUDA Toolkit (ajuste conforme canal)
    "cudnn"                        ; cuDNN (ajuste conforme canal)
    "coreutils"
    "findutils"
    "grep"
    "which"
    "rsync"
  ))
SCM

# Utilize guix time-machine se quiser pinar canais (padrão: usa os canais do usuário).
GUIX_TM=(guix shell -m "$MANIFEST_FILE" --check)

# --- Verificação de GPU dentro do ambiente Guix ---
echo "========================================================="
echo "[INFO] Verificando PyTorch + CUDA no ambiente Guix..."
"${GUIX_TM[@]}" -- python - <<'PY'
import torch, sys
print(f"PyTorch: {torch.__version__}")
avail = torch.cuda.is_available()
print(f"CUDA disponível: {avail}")
if avail:
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    sys.exit(10)
PY

# --- ETAPA 1: STAGING (copiar dados para $SCRATCH) ---
echo "========================================================="
echo "Iniciando Staging: Copiando dataset para o $SCRATCH..."

# 1) Sanidade na origem (HOME): precisa ter 200 classes no val e não ter val/images
if [ -d "$DATASET_SRC_HOME/val/images" ]; then
  echo "ERRO: Origem tem val/images (val está no formato original); reorganize no HOME antes."
  exit 11
fi
n_val_home=$(find "$DATASET_SRC_HOME/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n_val_home" -eq 200 ] || { echo "ERRO: Origem val tem $n_val_home dirs (esperado=200)."; exit 12; }

# 2) Cópia limpa para o SCRATCH
rm -rf "$DATASET_DST_SCRATCH"
rsync -a --delete "$DATASET_SRC_HOME/" "$DATASET_DST_SCRATCH/"

# 3) Sanidade no destino (SCRATCH)
if [ -d "$DATASET_DST_SCRATCH/val/images" ]; then
  echo "ERRO: val/images encontrado no SCRATCH (dataset inválido para ImageFolder)."
  exit 21
fi
n_val_scratch=$(find "$DATASET_DST_SCRATCH/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n_val_scratch" -eq 200 ] || { echo "ERRO: SCRATCH val tem $n_val_scratch dirs (esperado=200)."; exit 22; }

echo "Staging concluído e validado."
echo "========================================================="

# --- ETAPA 2: EXECUÇÃO COM DADOS LOCAIS E GPU ---
echo "Iniciando Job de Teste..."

DATASET_PATH_TRAIN="$DATASET_DST_SCRATCH/train"
DATASET_PATH_VAL="$DATASET_DST_SCRATCH/val"

# Alocador CUDA do PyTorch
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

# Opcional: se o cluster exigir, propague a libcuda do host (ex.: /usr/lib64/libcuda.so) e dispositivos /dev/nvidia*.
# Em muitos ambientes sem container, não é necessário; ajuste apenas se indicado pela doc do seu cluster. [web:13]
# export LD_PRELOAD="/usr/lib64/libcuda.so"

# Executa seu main dentro do ambiente Guix com CUDA
"${GUIX_TM[@]}" -- python -m src.main \
  --model_name "$1" \
  --config_path "$3" \
  --train_data_path "$DATASET_PATH_TRAIN" \
  --val_data_path "$DATASET_PATH_VAL" \
  --seed "$2"

echo "========================================================="
echo "JOB DE TESTE CONCLUÍDO (GUIX + GPU)"
echo "========================================================="
