#!/bin/sh
# submit_all_24h.sh - Submete todos os jobs de modelo com um tempo fixo de 24 horas.

set -eu
mkdir -p logs

# Função para submeter um job com tempo fixo
submit_model() {
  model="$1"
  seed="$2"
  time_limit="24:00:00"  # Tempo fixo de 24 horas para todos os modelos

  echo "--> Submetendo job: $model (seed $seed) | Tempo Limite: $time_limit"
  
 
  config_path="configs/${model}_config.yaml"

  # Verifica se o arquivo de config existe
  if [ ! -f "$config_path" ]; then
    echo "ERRO: Arquivo de configuração não encontrado para o modelo $model em $config_path"
    return 1
  fi

  sbatch --job-name="${model}_seed_${seed}" \
         --time="$time_limit" \
         run_all_experiment.slurm "$model" "$seed" "$config_path"
  
  sleep 1 # Pausa para não sobrecarregar o escalonador do Slurm
}

# Lista completa de modelos 
MODELS=" densenet121 efficientnet_b0 efficientnet_b7 inception_v3 inception_v4 mobilenet_V1 resnet34 resnet101 shufflenet_v2 squeezenet vgg16 xception"   
# Sementes para as execuções independentes
SEEDS="42 52 62"

# Loop para submeter um job para cada combinação de modelo e semente
for m in $MODELS; do
  for s in $SEEDS; do
    submit_model "$m" "$s"
  done
done

echo "========================================================="
echo "Todos os jobs foram submetidos com um tempo limite de 24 horas."
echo "========================================================="

