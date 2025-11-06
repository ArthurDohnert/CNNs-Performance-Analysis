;; --- manifest-cpu.scm ---
#:use-module (guix packages)
#:use-module (gnu packages python-science)
#:use-module (gnu packages python-xyz)
#:use-module (gnu packages graphics)
#:use-module (gnu packages machine-learning)
#:use-module (gnu packages maths)
#:use-module (gnu packages text-utils)
#:use-module (my-local-packages timm)

(specifications->manifest
  (list
   "python"
   "python-psutil"
   "python-pytorch" 
   "python-timm"    
   "python-numpy"
   "python-pandas"
   "python-scikit-learn"
   "python-matplotlib"
   "python-seaborn"
   "python-tqdm"
   "python-pillow"
   "python-pyyaml"
   "python-tabulate"
   ))