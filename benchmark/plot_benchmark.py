import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho para a pasta onde estão os CSVs
benchmark_dir = os.path.join("benchmark")
output_dir = "graficos_barras"
os.makedirs(output_dir, exist_ok=True)

# Configurações desejadas
configuracoes = [
    (3, 3, 7, "rapidfuzz"), (3, 3, 7, "sequence"),
    (3, 5, 5, "rapidfuzz"), (3, 5, 5, "sequence"),
    (3, 6, 4, "rapidfuzz"), (3, 6, 4, "sequence"),
    (5, 3, 7, "rapidfuzz"), (5, 3, 7, "sequence"),
    (5, 5, 5, "rapidfuzz"), (5, 5, 5, "sequence"),
    (5, 6, 4, "rapidfuzz"), (5, 6, 4, "sequence"),
    (10, 3, 7, "rapidfuzz"), (10, 3, 7, "sequence"),
    (10, 5, 5, "rapidfuzz"), (10, 5, 5, "sequence"),
    (10, 6, 4, "rapidfuzz"), (10, 6, 4, "sequence"),
]

# Lê todos os arquivos CSV de benchmark
csv_files = [f for f in os.listdir(benchmark_dir) if f.endswith(".csv") and f.startswith("benchmark")]
df_list = []

for file in csv_files:
    file_path = os.path.join(benchmark_dir, file)
    try:
        df = pd.read_csv(file_path)
        df_list.append(df)
    except Exception as e:
        print(f"[!] Erro ao ler {file}: {e}")

# Concatena todos os dados
if not df_list:
    raise FileNotFoundError("Nenhum arquivo CSV válido encontrado na pasta benchmark/")
df = pd.concat(df_list, ignore_index=True)

# Cria os gráficos de barras
for (top_k, peso_emb, peso_txt, matcher) in configuracoes:
    filtro = (
        (df["top_k"] == top_k) &
        (df["peso_emb"] == peso_emb / 10) &
        (df["peso_txt"] == peso_txt / 10) &
        (df["matcher"] == matcher)
    )
    df_config = df[filtro]
    if df_config.empty:
        continue

    # Pivot: perguntas no eixo X, modelos nas colunas
    pivot = df_config.pivot(index="pergunta", columns="modelo", values="sim_mean")
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(f"Similaridade Média por Pergunta - k{top_k} e={peso_emb} t={peso_txt} {matcher}")
    ax.set_ylabel("Similaridade média")
    ax.set_xlabel("Pergunta")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    nome_fig = f"barras_k{top_k}_e{peso_emb}_t{peso_txt}_{matcher}.png"
    plt.savefig(os.path.join(output_dir, nome_fig))
    plt.close()

print(f"✅ Gráficos salvos em: {output_dir}/")