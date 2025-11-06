;; timm.scm — pacote Python timm 1.0.21 para Guix
(define-module (my-local-packages timm)
  #:use-module (guix packages)
  #:use-module (guix download)
  #:use-module (guix build-system python)
  #:use-module (guix licenses)
  #:use-module (gnu packages python)
  #:use-module (gnu packages python-xyz)
  #:use-module (gnu packages image)
  #:use-module (gnu packages machine-learning))

(define-public python-timm
  (package
    (name "python-timm")
    (version "1.0.21")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "timm" version))
       (sha256
        (base32 "0bvhrdv95x27pppncmvx6x3w21hwlwmy4is6xgygim0vylmfhc93"))))
    (build-system python-build-system)
    (propagated-inputs
     (list python-torch
           python-numpy
           python-pillow
           python-requests))
    (home-page "https://github.com/huggingface/pytorch-image-models")
    (synopsis "PyTorch Image Models (timm)")
    (description
     "Biblioteca de modelos e utilitários para redes neurais em PyTorch, incluindo implementações de arquiteturas modernas e pré-treinadas.")
    (license asl2.0)))

