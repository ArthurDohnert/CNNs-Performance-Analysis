;; --- manifest-gpu.scm ---
(use-modules
  (guix packages)
  (guix profiles)
  (gnu packages python-science)
  (gnu packages python-xyz)
  (gnu packages graphics)
  (gnu packages machine-learning)
  (gnu packages maths)
  (my-local-packages timm))

(specifications->manifest
 (list
  "python"
  "python-psutil"
  "python-nvidia-ml-py"
  "python-pytorch-with-cuda11"
  "python-timm"
  "python-numpy"
  "python-pandas"
  "python-scikit-learn"
  "python-matplotlib"
  "python-seaborn"
  "python-tqdm"
  "python-pillow"
  "python-pyyaml"
  "python-tabulate"))
