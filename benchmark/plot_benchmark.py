import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]

cores = ['blue', 'orange']
PCA_COMPONENTE = 256  # Foco em PCA 256

dfs = []

for modelo in MODELOS:
    modelo_safe = modelo.replace("/", "_")
    arquivo = os.path.join(os.path.dirname(__file__), f'benchmark_{modelo_safe}_pca{PCA_COMPONENTE}.csv')
    if not os.path.exists(arquivo):
        print(f"Aviso: Arquivo {arquivo} não encontrado.")
        continue

    df = pd.read_csv(arquivo)
    df['modelo'] = modelo
    dfs.append(df)

if not dfs:
    print("Nenhum arquivo encontrado para gerar gráficos.")
    sys.exit()

df_all = pd.concat(dfs, ignore_index=True)

## 1. Gráfico: Tempo médio de consulta
tempo_medio = df_all.groupby('modelo')['tempo'].mean()

plt.figure(figsize=(8, 5))
plt.bar(tempo_medio.index, tempo_medio.values, color=cores)
plt.ylabel('Tempo Médio de Consulta (s)')
plt.title(f'Tempo Médio de Consulta (PCA = {PCA_COMPONENTE})')
plt.savefig(os.path.join(os.path.dirname(__file__), f'tempo_medio_pca{PCA_COMPONENTE}.png'))
plt.close()
print(f"Gráfico salvo: tempo_medio_pca{PCA_COMPONENTE}.png")

## 2. Gráfico: Similaridade Média
sim_mean = df_all.groupby('modelo')['sim_mean'].mean()

plt.figure(figsize=(8, 5))
plt.bar(sim_mean.index, sim_mean.values, color=cores)
plt.ylabel('Similaridade Média')
plt.title(f'Similaridade Média (PCA = {PCA_COMPONENTE})')
plt.savefig(os.path.join(os.path.dirname(__file__), f'sim_mean_pca{PCA_COMPONENTE}.png'))
plt.close()
print(f"Gráfico salvo: sim_mean_pca{PCA_COMPONENTE}.png")

## 3. Gráfico: Similaridade Máxima
sim_max = df_all.groupby('modelo')['sim_max'].mean()

plt.figure(figsize=(8, 5))
plt.bar(sim_max.index, sim_max.values, color=cores)
plt.ylabel('Similaridade Máxima')
plt.title(f'Similaridade Máxima (PCA = {PCA_COMPONENTE})')
plt.savefig(os.path.join(os.path.dirname(__file__), f'sim_max_pca{PCA_COMPONENTE}.png'))
plt.close()
print(f"Gráfico salvo: sim_max_pca{PCA_COMPONENTE}.png")

## 4. Gráfico: top_k vs sim_mean
plt.figure(figsize=(8, 5))
for idx, modelo in enumerate(MODELOS):
    df_m = df_all[df_all['modelo'] == modelo]
    df_group = df_m.groupby('top_k')['sim_mean'].mean().reset_index()
    plt.plot(df_group['top_k'], df_group['sim_mean'], marker='o', label=modelo, color=cores[idx])

plt.xlabel('top_k')
plt.ylabel('Similaridade Média')
plt.title(f'top_k vs Similaridade Média (PCA = {PCA_COMPONENTE})')
plt.legend()
plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), f'topk_vs_sim_mean_pca{PCA_COMPONENTE}.png'))
plt.close()
print(f"Gráfico salvo: topk_vs_sim_mean_pca{PCA_COMPONENTE}.png")

## 5. Gráfico: limiar vs sim_mean
plt.figure(figsize=(8, 5))
for idx, modelo in enumerate(MODELOS):
    df_m = df_all[df_all['modelo'] == modelo]
    df_group = df_m.groupby('limiar')['sim_mean'].mean().reset_index()
    plt.plot(df_group['limiar'], df_group['sim_mean'], marker='o', label=modelo, color=cores[idx])

plt.xlabel('Limiar')
plt.ylabel('Similaridade Média')
plt.title(f'Limiar vs Similaridade Média (PCA = {PCA_COMPONENTE})')
plt.legend()
plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), f'limiar_vs_sim_mean_pca{PCA_COMPONENTE}.png'))
plt.close()
print(f"Gráfico salvo: limiar_vs_sim_mean_pca{PCA_COMPONENTE}.png")

print("Todos os gráficos comparativos gerados com sucesso.")
