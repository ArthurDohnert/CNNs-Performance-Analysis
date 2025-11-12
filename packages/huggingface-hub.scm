;; packages/huggingface-hub.scm — Pacote Python 'huggingface-hub' para Guix
(define-module (packages huggingface-hub)
  #:use-module (guix packages)
  #:use-module (guix download)
  #:use-module (guix build-system python)
  #:use-module (guix licenses)
  #:use-module (gnu packages python)
  #:use-module (gnu packages python-build)
  #:use-module (gnu packages python-web)  ; Para python-httpx
  #:use-module (gnu packages python-xyz)) ; Para a maioria das dependências

(define-public python-huggingface-hub
  (package
    (name "python-huggingface-hub")
    ;; Uma versão recente e testada.
    (version "0.20.3")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "huggingface-hub" version))
       (sha256
        (base32
         "10f8y13l1l4f0m5s6c6k8i2kbsl11h2c72b88g4q3a15n92g8x8q"))))
    (build-system python-build-system)

    ;; Dependências corretas para huggingface-hub 0.20.3
    (propagated-inputs
     (list python-filelock            ; 
           python-fsspec             ; 
           python-httpx              ; 
           python-packaging          ; 
           python-pyyaml             ; 
           python-shellingham        ; 
           python-tqdm               ; 
           python-typer              ; 
           python-typing-extensions  ; 
           ))
    (home-page "https://github.com/huggingface/huggingface_hub")
    (synopsis "Client library for the Hugging Face Hub")
    (description
     "This library provides a set of tools to interact with the
Hugging Face Hub, a platform for sharing and discovering machine
learning models, datasets, and demos.")
    (license asl2.0)))