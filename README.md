### CNNs Performance Analysis ###
Repositório para executar e comparar modelos CNN em Tiny ImageNet-200, registrar métricas de treino/validação e gerar análises de desempenho (acurácia, tempo/latência e recursos).​

### Estrutura do projeto ###
- configs/ — arquivos de configuração dos experimentos (hiperparâmetros, dataset, caminhos de saída).​

- data/tiny-imagenet-200/ — dataset Tiny ImageNet-200 preparado para treino e validação.​

- logs/ — logs brutos de execução (stdout/stderr, CSV por época, métricas de GPU/CPU, slurm outputs).​

- logs_csv/ — logs já normalizados em CSV para agregação posterior.​

- results/ — saídas dos experimentos (checkpoints temporários, métricas por rodada).​

- src/ — código-fonte dos treinamentos, avaliação, utilitários e coleta de métricas.​

- tests/ — testes unitários/integração.​

- trained_models/ — checkpoints finais dos modelos treinados.​


scripts e raiz:

- prepare_tiny_imagenet_val.py — script para preparar/organizar a partição de validação do Tiny ImageNet.​

- run_experiment.slurm, run_all_experiment.slurm, run_all_guix.slurm, submit_all.sh, submit_test.sh, test.slurm — submissão/automação de jobs (SLURM/Guix).​

- logs_processor.ipynb — notebook principal para processar logs e gerar agregações/gráficos.​

- logs_processor_old.ipynb — versão anterior do pipeline de logs.​

- requirements.txt — dependências Python.​

- manifest.scm — manifesto do GNU Guix para ambientes reprodutíveis.​

### Instalação do ambiente ###

Opção 1 — Guix:

>>      guix shell -m manifest.scm 

Opção 2 — venv + pip:

>>      python -m venv venv​

>>      source venv/bin/activate (Linux/macOS) ou venv\Scripts\activate (Windows)​

>>      pip install -r requirements.txt​


### Preparação do dataset Tiny ImageNet-200 ###
Baixe o Tiny ImageNet-200 e extraia para data/tiny-imagenet-200/ mantendo a estrutura padrão (train, val, test).​

Execute: 
>>      python prepare_tiny_imagenet_val.py
para criar a pasta val/class e o annotations.txt conforme esperado pelos loaders.​

Estrutura esperada:

data/tiny-imagenet-200/train/<wnid>/images/*.JPEG​

data/tiny-imagenet-200/val/images/*.JPEG e val/val_annotations.txt (após script, val/<wnid>/images).​

### Como rodar uma run de um modelo ###

Execução local:

>>        python -m src.main 
>>        --model_name 
>>        --config_path 
>>        --train_data_path 
>>        --val_data_path
>>        --seed 


Via SLURM:

>>       sbatch run_experiment.slurm .​
>>         --model_name 
>>        --config_path 
>>        --train_data_path 
>>        --val_data_path
>>        --seed 

Saídas esperadas:

logs/<run_id>/ … arquivos .log, métricas por época, tempos, uso de memória.​

### Como rodar os experimentos ###

Definição dos modelos e das seeds no arquivo submit_all.sh

>>       sh submit_all.sh

### Como gerar análises com o logs_processor ###


Abra logs_processor.ipynb.​

Rode as células na ordem.


Convenções de configuração
configs/<nome>.yaml define: modelo (ex.: vgg16, resnet18), dataset root, batch size, epochs, otimizador, LR schedule, augmentations e opções de logging.​






