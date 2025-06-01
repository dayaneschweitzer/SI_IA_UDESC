import os
import pickle
import time
import textwrap
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Carrega modelo e dados
with open("dados_completos.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = data["embeddings"]
nome_arquivos = data["nomes_arquivos"]

# Monta índice
index = NearestNeighbors(n_neighbors=10, metric="cosine")
index.fit(embeddings)

# Modelo carregado externamente
modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def busca_semantica(pergunta, modelo, modelo_nome="paraphrase-multilingual-MiniLM-L12-v2"):
    pergunta_lower = pergunta.lower()
    inicio = time.time()
    pergunta_embedding = modelo.encode([pergunta], convert_to_numpy=True)
    distancias, indices = index.kneighbors(pergunta_embedding)
    resultados = []

    for idx in indices[0]:
        conteudo = chunks[idx]
        nome_arquivo = nome_arquivos[idx]

        # Busca a melhor linha parecida dentro do chunk
        melhor_linha = ""
        melhor_score = 0.0
        for linha in conteudo.splitlines():
            score = SequenceMatcher(None, pergunta_lower, linha.lower()).ratio()
            if score > melhor_score:
                melhor_score = score
                melhor_linha = linha

        trecho = melhor_linha.strip() if melhor_linha else textwrap.shorten(conteudo.replace("\n", " "), width=300)
        distancia = distancias[0][list(indices[0]).index(idx)]
        similaridade_base = round(1 - (distancia / 2), 4)

        # Ajuste com mais peso textual e bônus por termos da pergunta no chunk
        peso_embedding = 0.5
        peso_texto = 0.5
        similaridade = round(peso_embedding * similaridade_base + peso_texto * melhor_score, 4)

        # Bônus por presença explícita de termos da pergunta no chunk
        tokens_pergunta = set(pergunta_lower.split())
        tokens_chunk = set(conteudo.lower().split())
        intersecao = tokens_pergunta.intersection(tokens_chunk)

        if len(intersecao) >= 2:
            similaridade = min(similaridade + 0.05, 1.0)
        if melhor_score > 0.92:
            similaridade = min(similaridade + 0.03, 1.0)

        resultados.append({
            "nome": nome_arquivo,
            "trecho": trecho,
            "link": "#",  # Pode ser ajustado se houver mapeamento
            "similaridade": similaridade
        })

    # Ordena os resultados pela nova similaridade combinada
    resultados.sort(key=lambda x: x["similaridade"], reverse=True)
    tempo = round(time.time() - inicio, 3)
    return resultados[:5]


def busca_literal_em_todos(pergunta, limite=0.4):
    resultados = []
    pergunta_lower = pergunta.lower()

    for i, conteudo in enumerate(chunks):
        nome_arquivo = nome_arquivos[i]
        score = SequenceMatcher(None, conteudo.lower(), pergunta_lower).ratio()
        if score >= limite:
            trecho = textwrap.shorten(conteudo.replace("\n", " "), width=400, placeholder=" [...]")
            resultados.append({
                "nome": nome_arquivo,
                "trecho": trecho,
                "link": "#",
                "similaridade": round(score, 2)
            })

    resultados.sort(key=lambda x: x["similaridade"], reverse=True)
    return resultados[:5]