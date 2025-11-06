;; timm.scm — pacote Python timm 1.0.21 para Guix
(use-modules (guix packages)
             (guix download)
             (guix build-system python)
             (guix licenses)
             (gnu packages python)
             (gnu packages python-xyz)
             (gnu packages image)
             (gnu packages machine-learning))

(package
  (name "python-timm")
  (version "1.0.21")
  (source
   (origin
     (method url-fetch)
     (uri (pypi-uri "timm" version))
     (sha256
      (base32 "0bvhrdv95x27pppncmvx6x3w21hwlwmy4is6xgygim0vylmfhc93")))) ; verificado no PyPI
  (build-system python-build-system)
  (propagated-inputs
   (list python-torch                ; já está no teu ambiente
         python-numpy
         python-pillow
         python-requests))
  (home-page "https://github.com/huggingface/pytorch-image-models")
  (synopsis "PyTorch Image Models (timm)")
  (description
   "Biblioteca de modelos e utilitários para redes neurais em PyTorch, incluindo implementações de arquiteturas modernas e pré-treinadas.")
  (license asl2.0))
