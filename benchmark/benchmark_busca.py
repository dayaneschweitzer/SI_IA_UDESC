
import os
import sys
sys.path.append(os.path.abspath(".."))

import time
import pickle
import itertools
import numpy as np
import pandas as pd
from buscar import busca_semantica
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Carrega os dados base
with open("../dados_completos.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
nome_arquivos = data["nomes_arquivos"]

PERGUNTAS_TESTE = [
    "Aprova alteração curricular do Curso de Mestrado Acadêmico em Computação Aplicada",
    "Portaria MEC 609",
    "Resolução do PPGCAP sobre credenciamento de docentes",
    "Reconhecimento do curso de Pós-graduação em Computação Aplicada",
    "Portaria MEC 656/2017",
    "Resolução 037",
    "Qual a política de ensino médio da UDESC?",
    "Diretrizes da ONU sobre IA",
    "Regras para bolsas de estudo internacionais",
    "Clima em Florianópolis"
]

MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]

TOP_K_LIST = [3, 5, 10]
PESOS = [(0.5, 0.5), (0.6, 0.4), (0.3, 0.7)]
MATCHERS = ["sequence", "rapidfuzz"]

os.makedirs("benchmark", exist_ok=True)

for modelo_nome in MODELOS:
    print(f"[Modelo] {modelo_nome}")
    modelo = SentenceTransformer(modelo_nome)
    embeddings = modelo.encode(chunks, convert_to_numpy=True)

    for top_k, (peso_emb, peso_txt), metric in itertools.product(TOP_K_LIST, PESOS, MATCHERS):
        index = NearestNeighbors(n_neighbors=top_k, metric="cosine")
        index.fit(embeddings)
        resultados = []

        for pergunta in PERGUNTAS_TESTE:
            try:
                resposta, tempo = busca_semantica(
                    pergunta,
                    modelo=modelo,
                    modelo_nome=modelo_nome,
                    top_k=top_k,
                    peso_embedding=peso_emb,
                    peso_texto=peso_txt,
                    metric=metric,
                    index=index
                )

            except Exception as e:
                print(f"[!] Erro ao buscar com modelo {modelo_nome}: {e}")
                continue

            sim_max = max([r['similaridade'] for r in resposta], default=0)
            sim_mean = np.mean([r['similaridade'] for r in resposta]) if resposta else 0

            resultados.append({
                'modelo': modelo_nome,
                'top_k': top_k,
                'peso_emb': peso_emb,
                'peso_txt': peso_txt,
                'matcher': metric,
                'pergunta': pergunta,
                'respostas': len(resposta),
                'sim_max': sim_max,
                'sim_mean': sim_mean,
                'tempo': tempo
            })

        df = pd.DataFrame(resultados)
        nome_modelo = modelo_nome.replace("/", "_")
        nome_csv = f"benchmark_{nome_modelo}_k{top_k}_e{int(peso_emb*10)}_t{int(peso_txt*10)}_{metric}.csv"
        df.to_csv(os.path.join("benchmark", nome_csv), index=False)
        print(f"[OK] Benchmark salvo: {nome_csv}")
