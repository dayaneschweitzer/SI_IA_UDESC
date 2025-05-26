import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import pandas as pd
import numpy as np
from buscar import busca_semantica
from sentence_transformers import SentenceTransformer

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

TOP_K_LIST = [5, 10, 20]
LIMIAR_LIST = [0.6, 0.7, 0.75, 0.8]
MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]
PCA_COMPONENTES = [256, 512, 768, 1024]

for modelo_nome in MODELOS:
    modelo_safe = modelo_nome.replace("/", "_")
    modelo_teste = SentenceTransformer(modelo_nome)

    for n_pca in PCA_COMPONENTES:
        suf = f"{modelo_safe}_pca{n_pca}"

        # Verifica se o índice e o PCA model existem
        idx_file = f'index_faiss_{suf}.idx'
        pca_file = f'pca_model_{suf}.pkl'

        if not os.path.exists(idx_file) or not os.path.exists(pca_file):
            print(f"PCA {n_pca} não disponível para {modelo_safe}, pulando...")
            continue

        print(f"Testando modelo: {modelo_nome} | PCA: {n_pca}")

        resultados = []

        for top_k, limiar in itertools.product(TOP_K_LIST, LIMIAR_LIST):
            for pergunta in PERGUNTAS_TESTE:
                try:
                    resposta, tempo = busca_semantica(
                        pergunta=pergunta,
                        modelo=modelo_teste,
                        modelo_nome=modelo_nome,
                        top_k=top_k,
                        limiar=limiar,
                        n_pca=n_pca 
                    )
                except Exception as e:
                    print(f"Erro ao buscar: {e}")
                    continue

                n_respostas = len(resposta)
                sim_max = max([r['similaridade'] for r in resposta], default=0)
                sim_mean = np.mean([r['similaridade'] for r in resposta]) if resposta else 0

                resultados.append({
                    'modelo': modelo_nome,
                    'pca': n_pca,
                    'top_k': top_k,
                    'limiar': limiar,
                    'pergunta': pergunta,
                    'respostas': n_respostas,
                    'sim_max': sim_max,
                    'sim_mean': sim_mean,
                    'tempo': tempo
                })

        df_resultados = pd.DataFrame(resultados)

        output_csv = os.path.join(
            os.path.dirname(__file__), 
            f'benchmark_{modelo_safe}_pca{n_pca}.csv'
        )
        df_resultados.to_csv(output_csv, index=False)
        print(f"Benchmark salvo como {output_csv}")
