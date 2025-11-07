;; packages/timm.scm — Pacote Python 'timm' 1.0.21 para Guix (Corrigido)
(define-module (timm)
  #:use-module (guix packages)
  #:use-module (guix download)
  #:use-module (guix build-system python)
  #:use-module (guix licenses)

  ;; Módulos necessários para as *novas* dependências corretas:
  #:use-module (gnu packages python-xyz)
  ;; -> Para python-pyyaml 
  #:use-module (gnu packages machine-learning)
  ;; -> Para python-torchvision  e python-safetensors 
  #:use-module (guix-science-nonfree packages machine-learning)
  ;; -> Para python-pytorch-with-cuda11 

  ;; Importa o nosso novo pacote local 'huggingface-hub'
  ;; Este é o passo-chave.
  #:use-module (huggingface-hub))

(define-public python-timm
  (package
    (name "python-timm")
    (version "1.0.21")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "timm" version))
       ;; O hash SHA256 do usuário estava correto.
       (sha256
        (base32 "0bvhrdv95x27pppncmvx6x3w21hwlwmy4is6xgygim0vylmfhc93"))))
    (build-system python-build-system)

    ;; 'propagated-inputs' corrigidos com base na pesquisa 
    (propagated-inputs
     (list
      ;; Dependências corretas para timm 1.0.21
      python-pytorch-with-cuda11  ; (torch)
      python-torchvision          ; (torchvision)
      python-pyyaml               ; (pyyaml)
      python-safetensors          ; (safetensors)
      python-huggingface-hub      ; (huggingface_hub)
      ))

    (home-page "https://github.com/huggingface/pytorch-image-models")
    (synopsis "PyTorch Image Models (timm)")
    (description
     "Uma biblioteca de modelos e utilitários para redes neurais em PyTorch,
incluindo implementações de arquiteturas modernas e pré-treinadas.")
    (license asl2.0)))