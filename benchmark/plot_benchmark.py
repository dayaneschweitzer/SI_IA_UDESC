import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def gerar_graficos():
    modelos = [
        'paraphrase-multilingual-MiniLM-L12-v2',
        'all-mpnet-base-v2'
    ]

    dfs = []
    for modelo in modelos:
        csv_path = os.path.join(os.path.dirname(__file__), f'benchmark_{modelo.replace("/", "_")}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
        else:
            print(f"Aviso: Arquivo {csv_path} não encontrado.")

    if not dfs:
        print("Nenhum arquivo de benchmark encontrado para gerar gráficos.")
        return

    df = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'plots'), exist_ok=True)

    ## --- Gráfico comparativo de linhas --- ##
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x='top_k', 
        y='sim_mean', 
        hue='modelo', 
        style='limiar', 
        markers=True,
        dashes=False
    )
    plt.title('Comparativo: Similaridade Média por Modelo, Top_K e Limiar')
    plt.xlabel('Top_K')
    plt.ylabel('Similaridade Média')
    plt.legend(title='Modelo / Limiar')
    plt.grid(True)

    plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', 'comparativo_modelos_limiar_topk.png'))
    plt.show()

    ## --- Heatmaps por modelo --- ##
    for modelo in modelos:
        csv_path = os.path.join(os.path.dirname(__file__), f'benchmark_{modelo.replace("/", "_")}.csv')
        if not os.path.exists(csv_path):
            continue

        df_modelo = pd.read_csv(csv_path)
        pivot = df_modelo.pivot_table(
            index='top_k',
            columns='limiar',
            values='sim_mean',
            aggfunc='mean'
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap='viridis')
        plt.title(f'Heatmap: Similaridade Média - {modelo}')
        plt.xlabel('Limiar')
        plt.ylabel('Top_K')

        plt.savefig(os.path.join(os.path.dirname(__file__), 'plots', f'heatmap_{modelo.replace("/", "_")}.png'))
        plt.show()

    print("✅ Gráficos gerados e salvos na pasta 'plots'.")

