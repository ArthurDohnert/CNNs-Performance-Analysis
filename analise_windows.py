import os
import subprocess
import sys

PYTHON_EXEC = sys.executable

def run_command(command, description):
    print(f"--- Iniciando: {description} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"ERRO ao executar: {description}")
        sys.exit(1) 
    print(f"--- Sucesso: {description} ---\n")

run_command(f'"{PYTHON_EXEC}" -m jupyter nbconvert --to notebook --execute --inplace logs_processor.ipynb', "Logs Processor Notebook")
run_command(f'"{PYTHON_EXEC}" -m jupyter nbconvert --to notebook --execute --inplace analysis_report.ipynb', "Analysis Report Notebook")

tex_files = ["reports/main.tex", "reports/slides.tex"]

for tex in tex_files:
    if not os.path.exists(tex):
        print(f"ERRO CRÍTICO: Arquivo não encontrado: {tex}")
        sys.exit(1)

    cmd = f"latexmk -pdf -cd -interaction=nonstopmode {tex}"
    
    run_command(cmd, f"Compilando {tex}")