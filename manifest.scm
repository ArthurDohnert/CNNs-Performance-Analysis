;; manifest.scm
(use-modules (guix)
             (gnu packages python-xyz)
             (gnu packages machine-learning)
             (gnu packages data-science)
             (gnu packages plotting)
             (gnu packages cuda)
             (gnu packages compression))

;; Lista completa de pacotes para o projeto de análise de performance de CNNs
(specification->manifest
  '(;; Core
    "python"
    "python-psutil"
    "python-pynvml"
    
    ;; PyTorch e CUDA
    "pytorch-cudatoolkit"
    "pytorch-cuda"
    "python-torchvision"
    
    ;; Análise e Manipulação de Dados
    "python-numpy"
    "python-pandas"
    "python-scikit-learn"
    
    ;; Visualização e Utilitários
    "python-matplotlib"
    "python-seaborn"
    "python-tqdm"
    "python-pillow"
    "python-yaml"
    "python-tabulate"))
