#!/bin/sh
# Submete jobs via /bin/sh, definindo o tempo por modelo e passando para o sbatch.

set -eu

mkdir -p logs

# Função para escolher tempo por modelo (HH:MM:SS)
submit_model() {
  model="$1"
  seed="$2"

  case "$model" in
    MobileNetV1|mobilenet_v1) time_limit="02:00:00" ;;
    VGG16|vgg16)              time_limit="06:00:00" ;;
    ResNet101|resnet101)      time_limit="15:00:00" ;;
    *)                        time_limit="04:00:00" ;;
  esac

  echo "Submetendo: $model seed $seed | tempo $time_limit"
  sbatch --job-name="${model}_seed_${seed}" \
         --time="$time_limit" \
         run_all_experiment.slurm "$model" "$seed"
}

# Modelos e sementes (strings simples para compatibilidade POSIX)
SEEDS="42 52 62"
MODELS="MobileNetV1 VGG16 ResNet101"

for m in $MODELS; do
  for s in $SEEDS; do
    submit_model "$m" "$s"
    sleep 1
  done
done

echo "Todos os jobs submetidos."
