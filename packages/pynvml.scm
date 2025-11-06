(define-module (my-local-packages pynvml)
  #:use-module (guix packages)
  #:use-module (guix download)
  #:use-module (guix build-system python)
  #:use-module (guix licenses)
  #:use-module (gnu packages python))

(define-public python-pynvml
  (package
    (name "python-pynvml")
    (version "11.5.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "nvidia-ml-py3" version))
       (sha256
        (base32 "0chxprnd4sxccm6z9p3i1g7pkdrh9bczl4vypqlzh8pyg1h7iv5v"))))
    (build-system python-build-system)
    (home-page "https://github.com/NVIDIA/pynvml")
    (synopsis "Python bindings for the NVIDIA Management Library (NVML)")
    (description "This package provides Python bindings for the NVIDIA Management Library (NVML), which provides access to GPU metrics such as temperature, power usage, and memory utilization.")
    (license asl2.0)))
