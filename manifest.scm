;; manifest.scm — Ambiente Guix para PyTorch com GPU (CUDA 11.8)
;; Requer canais não‑livres (ex.: guix-science-nonfree / guix-hpc-non-free / nonguix)
;; para disponibilizar CUDA/cuDNN/NCCL; ajuste os nomes/versões conforme o canal. 

(use-modules (guix profiles))

(specifications->manifest
 '(
   ;; Python base + utilitários
   "python@3.10"
   "python-pip"
   "python-setuptools"
   "python-wheel"
   "coreutils"
   "findutils"
   "grep"
   "which"
   "rsync"
   "git"

   ;; BLAS/LAPACK para NumPy/Scikit-Learn (wheels podem usar, ou fallback)
   "openblas"
   "lapack"

   ;; Dependências de imagem comuns ao torchvision/Pillow
   "libjpeg-turbo"
   "libpng"

   ;; PILHAS GPU — dos canais não‑livres (ajuste nomes/versões conforme canal)
   ;; Em muitos canais, "cuda-toolkit" permite @11.8; alguns usam "cuda" em vez de "cuda-toolkit".
   "cuda-toolkit@11.8"
   "cudnn"          ; geralmente cuDNN 8.x para CUDA 11.8
   "nccl"           ; NCCL compatível com CUDA 11.x
 ))
