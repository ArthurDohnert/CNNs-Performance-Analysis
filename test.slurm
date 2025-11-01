#!/bin/bash
#SBATCH --job-name=exemplo1
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Iniciando o job de exemplo..."  
guix describe
