#!/bin/bash
#SBATCH --job-name=exemplo1
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --get-user-env

set -euo pipefail

# Se Guix é perfil do usuário:
if [ -f "$HOME/.config/guix/current/etc/profile" ]; then
  . "$HOME/.config/guix/current/etc/profile"
fi
if [ -f "$HOME/.guix-profile/etc/profile" ]; then
  . "$HOME/.guix-profile/etc/profile"
fi

# Se Guix é via módulo:
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
  module --force purge || true
  module load guix || true
fi

which guix || { echo "guix não encontrado no PATH do job"; exit 127; }
echo "Iniciando o job de exemplo..."
guix describe
