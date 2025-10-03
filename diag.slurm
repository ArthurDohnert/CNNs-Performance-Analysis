#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --partition=shared
#SBATCH --time=00:05:00
#SBATCH --output=$HOME/diag_%j.out
#SBATCH --error=$HOME/diag_%j.err

set -x
echo "=== ENV START ==="
echo "SHELL=$SHELL"
echo "PATH=$PATH"
env | sort | sed -n '1,120p'
echo "=== BIN CHECK ==="
command -v module || echo "module: NOT FOUND"
command -v ml || echo "ml: NOT FOUND"
command -v modulecmd || echo "modulecmd: NOT FOUND"
command -v guix || echo "guix: NOT FOUND"

# Tente inicializar módulos por caminhos típicos do nó
for f in /usr/share/Modules/init/bash /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
  if [ -f "$f" ]; then
    echo "Sourcing $f"; source "$f"
    break
  fi
done

echo "=== AFTER INIT ==="
command -v module || echo "module: NOT FOUND"
module --version 2>/dev/null || true
module avail 2>&1 | head -n 50 || true
command -v guix || echo "guix: NOT FOUND"
ls -d /gnu/store/*/bin/guix 2>/dev/null | head -n 3 || true
