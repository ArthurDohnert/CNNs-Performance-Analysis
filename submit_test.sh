#!/bin/bash

# Script para submeter um único job de teste rápido para o MobileNetV1.

MODEL_TO_TEST="vgg16"
TEST_SEED=42 # Uma semente aleatória para o teste

echo "Submetendo job de teste para o modelo: ${MODEL_TO_TEST} com seed ${TEST_SEED}"

sbatch \
    --job-name="${MODEL_TO_TEST}_sanity_test" \
    run_pcad_test.slurm "${MODEL_TO_TEST}" "${TEST_SEED}"

echo "Job de teste submetido. Use 'squeue -u ehdmenezes' para monitorar."
