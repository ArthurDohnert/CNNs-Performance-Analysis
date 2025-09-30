#!/bin/bash

echo "========================================================="
echo "INICIANDO SUBMISSÃO DE TODOS OS JOBS DE TREINAMENTO"
echo "========================================================="

# Garante que o diretório de logs existe
mkdir -p logs

# Lista completa de modelos para o estudo de performance
# (Use os nomes exatos esperados pelo seu 'get_model')
ALL_MODELS=(
    # --- Lightweight ---
    "MobileNetV1"
    "SqueezeNet"
    "ShuffleNetV2"
    "EfficientNetB0"
    
    # --- Medium-Weight ---
    "VGG16"
    "ResNet34"
    "InceptionV3"
    "DenseNet121"
    
    # --- Heavyweight ---
    "ResNet101"
    "InceptionV4"
    "Xception"
    "EfficientNetB7"
)

# Loop para submeter um job para cada modelo
for model in "${ALL_MODELS[@]}"
do
    job_name="${model}_perf_test"
    echo "-> Submetendo job: ${job_name}"
    
    # Comando sbatch:
    # --job-name: Define um nome único para fácil identificação no 'squeue'
    # Passa o nome do modelo como o primeiro argumento ($1) para o run_experiment.slurm
    sbatch --job-name="${job_name}" run_experiment.slurm "${model}"
    
    # Pequena pausa para não sobrecarregar o escalonador do Slurm
    sleep 1
done

echo "========================================================="
echo "Todos os jobs foram submetidos."
echo "Use 'squeue -u <seu_usuario>' para monitorar o status."
echo "========================================================="
