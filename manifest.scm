(use-modules (guix profiles))

(specifications->manifest
 '(
   ;; Python + ferramentas para instalar via pip no ambiente puro
   "bash-minimal"
   "python@3.10"
   "python-pip"
   "python-setuptools"
   "python-wheel"

   ;; Utilitários usados no job
   "coreutils" "findutils" "grep" "which" "rsync" "git"

   ;; BLAS/LAPACK e libs de imagem úteis para NumPy/Pillow
   "openblas" "lapack"
   "libjpeg-turbo" "libpng"
 ))
