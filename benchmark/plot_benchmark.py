import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cria pasta de saída
os.makedirs('benchmark/plots', exist_ok=True)

# Modelos a serem processados
MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]

dfs = []

# Carregar CSVs de cada modelo
for modelo_nome in MODELOS:
    csv_path = os.path.join('benchmark', f'benchmark_{modelo_nome.replace("/", "_")}.csv')
    if not os.path.exists(csv_path):
        print(f"Aviso: Arquivo {csv_path} não encontrado.")
        continue
    df = pd.read_csv(csv_path)
    dfs.append(df)

if not dfs:
    print("Nenhum CSV encontrado. Abortando.")
    exit()

# Junta os DataFrames
df = pd.concat(dfs, ignore_index=True)

## --- Gráfico 1: Comparativo de linhas (modelo + limiar) ---

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

plt.savefig('benchmark/plots/comparativo_modelos_limiar_topk.png')
plt.close()

## --- Gráfico 2: Heatmaps por modelo ---

for modelo_nome in MODELOS:
    df_modelo = df[df['modelo'] == modelo_nome]
    pivot = df_modelo.pivot_table(
        index='top_k',
        columns='limiar',
        values='sim_mean',
        aggfunc='mean'
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='viridis')
    plt.title(f'Heatmap: Similaridade Média - {modelo_nome}')
    plt.xlabel('Limiar')
    plt.ylabel('Top_K')

    output_path = f'benchmark/plots/heatmap_{modelo_nome.replace("/", "_")}.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap salvo em: {output_path}")

print("Gráficos gerados com sucesso!")
