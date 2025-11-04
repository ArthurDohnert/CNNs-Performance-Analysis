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


   "openblas"
   "lapack"

   "libjpeg-turbo"
   "libpng"

   "cuda-toolkit@11.8"
   "cudnn"          ; geralmente cuDNN 8.x para CUDA 11.8
   "nccl"           ; NCCL compatível com CUDA 11.x
 ))
