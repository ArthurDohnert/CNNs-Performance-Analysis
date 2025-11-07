;; --- manifest-cpu.scm ---
(use-modules (guix profiles)
             (guix packages)
             (gnu packages python)
             (gnu packages python-science)
             (gnu packages python-xyz)
             (gnu packages graphics)
             (gnu packages machine-learning)
             (gnu packages maths)
             (timm)
             (huggingface-hub))

(packages->manifest
 (list python
       python-psutil
       python-pytorch
       python-timm
       python-numpy
       python-pandas
       python-scikit-learn
       python-matplotlib
       python-seaborn
       python-tqdm
       python-pillow
       python-pyyaml
       python-tabulate))