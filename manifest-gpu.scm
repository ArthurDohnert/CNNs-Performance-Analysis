;; --- manifest-gpu.scm ---

(add-to-load-path "/home/users/ehdmenezes/CNNs-Performance-Analysis/packages")


(use-modules (guix profiles)
             (guix packages)
             (gnu packages python)
             (gnu packages python-science)
             (gnu packages python-xyz)
             (gnu packages graphics)
             (gnu packages machine-learning)
             (gnu packages maths)
             (pynvml)
             (timm)
             (guix-science-nonfree packages cuda))

(packages->manifest
 (list python
       python-psutil
       python-pynvml     ; símbolo exportado por (pynvml)
       python-pytorch-with-cuda11
       python-timm       ; símbolo exportado por (timm)
       python-numpy
       python-pandas
       python-scikit-learn
       python-matplotlib
       python-seaborn
       python-tqdm
       python-pillow
       python-pyyaml
       python-tabulate))
