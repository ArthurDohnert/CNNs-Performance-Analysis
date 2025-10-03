#!/bin/bash

echo "========================================================="
echo "INICIANDO SUBMISSÃO DE JOBS COM TEMPO DE EXECUÇÃO DINÂMICO"
echo "========================================================="

mkdir -p logs

# --- MAPA DE TEMPOS DE EXECUÇÃO (Formato HH:MM:SS) ---
# **AÇÃO NECESSÁRIA**: Ajuste estes tempos com base na sua experiência
declare -A MODEL_EXECUTION_TIMES
MODEL_EXECUTION_TIMES=(
    # --- Lightweight Models (Ex: 2 horas) ---
    ["MobileNetV1"]="02:00:00"
    #["SqueezeNet"]="02:00:00"
    #["ShuffleNetV2"]="02:00:00"
    #["EfficientNetB0"]="03:00:00"

    # --- Medium-Weight Models (Ex: 6 horas) ---
    ["VGG16"]="06:00:00"
    #["ResNet34"]="06:00:00"
    #["InceptionV3"]="08:00:00"
    #["DenseNet121"]="07:00:00"
    
    # --- Heavyweight Models (Ex: 12-20 horas) ---
    ["ResNet101"]="15:00:00"
    #["InceptionV4"]="18:00:00"
    #["Xception"]="16:00:00"
    #["EfficientNetB7"]="20:00:00"
)

# Sementes aleatórias para as execuções independentes
SEEDS=(10 20 30)

# Loop para submeter um job para cada combinação de modelo e semente
for model in "${!MODEL_EXECUTION_TIMES[@]}"; do
    # Obtém o tempo de execução do mapa
    execution_time=${MODEL_EXECUTION_TIMES[$model]}

    for seed in "${SEEDS[@]}"; do
        job_name="${model}_seed_${seed}"
        echo "-> Submetendo job: ${job_name} | Tempo Limite: ${execution_time}"
        
        # Passa o tempo dinamicamente para o sbatch usando a flag --time
        sbatch \
            --job-name="${job_name}" \
            --time="${execution_time}" \
            run_experiment.slurm "${model}" "${seed}"
        
        sleep 1 # Evita sobrecarregar o escalonador
    done
done

echo "========================================================="
echo "Todos os jobs foram submetidos com tempos de execução personalizados."
echo "========================================================="
