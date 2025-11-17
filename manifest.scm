;; --- manifest-pip.scm ---

(use-modules (guix profiles)
             (guix packages)
             (gnu packages python)
             (gnu packages python-science)
             (gnu packages python-xyz)
             (gnu packages graphics)
             (gnu packages machine-learning)
             (gnu packages maths)
             (gnu packages bash)
             (gnu packages package-management)
             (gnu packages python-build)) 

(packages->manifest
 (list python
       python-pip            
       bash                 
       python-psutil
       python-numpy
       python-pandas
       python-scikit-learn
       python-matplotlib
       python-seaborn
       python-tqdm
       python-pillow
       python-pyyaml
       python-tabulate
       ))