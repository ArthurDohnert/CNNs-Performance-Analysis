# src/pipelines/analysis_pipeline.py

###
### Functions that handle the analysis and visualization of results
###

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def load_results_data(csv_path: str) -> pd.DataFrame:
    """Loads experiment results from a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo de resultados '{csv_path}' não foi encontrado.")
        return None

def plot_accuracy_vs_latency(df: pd.DataFrame, output_dir: str):
    """Plots accuracy vs. inference latency, colored by model class."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="elapsed_time_seconds",
        y="accuracy",
        hue="model_class",  # Assumindo que você tem essa coluna
        size="vram_usage_bytes",
        style="hardware_type", # Assumindo que você tem essa coluna
        sizes=(50, 500),
        alpha=0.7
    )
    
    # Adiciona anotações com o nome do modelo
    for i in range(df.shape[0]):
        plt.text(
            x=df.elapsed_time_seconds[i] + 0.01,
            y=df.accuracy[i],
            s=df.model_name[i],
            fontdict=dict(color='black', size=8)
        )

    plt.title('Acurácia vs. Latência de Inferência', fontsize=16)
    plt.xlabel('Latência de Inferência (segundos)', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Classe do Modelo / Hardware')
    
    plot_path = os.path.join(output_dir, 'accuracy_vs_latency.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")

def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generates and saves a summary table of the results."""
    summary = df.groupby(['model_class', 'model_name']).agg({
        'accuracy': 'max',
        'elapsed_time_seconds': 'mean',
        'vram_usage_bytes': 'mean'
    }).sort_values(by='accuracy', ascending=False)
    
    summary_path = os.path.join(output_dir, 'summary_results.md')
    summary.to_markdown(summary_path)
    print(f"Tabela de resumo salva em: {summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Análise de Resultados")
    parser.add_argument('--results_file', type=str, required=True, help='Caminho para o arquivo CSV com os resultados consolidados.')
    parser.add_argument('--output_dir', type=str, default='reports', help='Diretório para salvar os relatórios e gráficos.')
    
    args = parser.parse_args()
    
    # Cria o diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Carrega os dados
    results_df = load_results_data(args.results_file)
    
    if results_df is not None:
        # Gera as visualizações e relatórios
        plot_accuracy_vs_latency(results_df, args.output_dir)
        generate_summary_table(results_df, args.output_dir)
        print("Análise concluída.")
