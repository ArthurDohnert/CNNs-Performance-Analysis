#!/bin/bash
#SBATCH --job-name=exemplo1
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Iniciando o job de exemplo..."  
guix describe
