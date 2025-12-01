#!/bin/bash

set -e

echo "--- Iniciando: Logs Processor Notebook ---"
jupyter nbconvert --to notebook --execute --inplace logs_processor.ipynb
echo "--- Sucesso: Logs Processor Notebook ---"
echo ""

echo "--- Iniciando: Analysis Report Notebook ---"
jupyter nbconvert --to notebook --execute --inplace analysis_report.ipynb
echo "--- Sucesso: Analysis Report Notebook ---"
echo ""


TEX_FILES=("reports/main.tex" "reports/slides.tex")

for tex in "${TEX_FILES[@]}"; do
    echo "--- Compilando: $tex ---"
    
    if [ ! -f "$tex" ]; then
        echo "ERRO CRÍTICO: Arquivo não encontrado: $tex"
        exit 1
    fi

    latexmk -pdf -cd -interaction=nonstopmode "$tex"
    
    echo "--- Sucesso: $tex ---"
    echo ""
done

echo ">>> PROCESSO CONCLUÍDO COM SUCESSO! <<<"