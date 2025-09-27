(use-modules (guix)
             (gnu packages python-xyz)
             (gnu packages machine-learning)
             (gnu packages data-science) ; Módulo para numpy/pandas
             (gnu packages plotting)     ; Módulo para matplotlib
             (gnu packages cuda))

(specification->manifest
  '("python"
    "python-psutil"
    "pytorch-cudatoolkit"
    "pytorch-cuda"
    "python-torchvision"
    "python-numpy"
    "python-pandas"
    "python-matplotlib"
    "python-pillow"))
