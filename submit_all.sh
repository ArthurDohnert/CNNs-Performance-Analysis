#!/bin/bash

# Crie um diretório para os logs, se não existir
mkdir -p logs

# Lista de modelos a serem testados
MODELS=(
    "MobileNetV2"
    "SqueezeNet"
    "EfficientNetB0"
    "VGG16"
    "ResNet34"
    "DenseNet121"
    "ResNet101"
    "Xception"
    "EfficientNetB7"
)

# Loop para submeter um job para cada modelo
for model_name in "${MODELS[@]}"
do
    echo "Submetendo job para o modelo: $model_name"
    sbatch --job-name="$model_name" run_experiment.slurm "$model_name"
    sleep 1 # Pequena pausa para não sobrecarregar o escalonador do Slurm
done

echo "Todos os jobs foram submetidos."
