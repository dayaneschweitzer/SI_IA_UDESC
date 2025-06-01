import os
import pickle
import time
import textwrap
import re
import unicodedata
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# Carrega dados
with open("dados_completos.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = data["embeddings"]
nome_arquivos = data["nomes_arquivos"]

# Monta índice semântico
index = NearestNeighbors(n_neighbors=10, metric="cosine")
index.fit(embeddings)

modelo = SentenceTransformer("all-mpnet-base-v2")

def normalizar(texto):
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"[\s_./-]+", "", texto)

def encontrar_link(nome):
    base = nome.split("_chunk")[0] if "_chunk" in nome else nome
    return f"/documento/{base}"

def busca_semantica(pergunta, modelo, modelo_nome="all-mpnet-base-v2"):
    pergunta_lower = pergunta.lower()
    pergunta_embedding = modelo.encode([pergunta], convert_to_numpy=True)
    distancias, indices = index.kneighbors(pergunta_embedding)
    resultados = []

    for idx in indices[0]:
        conteudo = chunks[idx]
        nome_arquivo = nome_arquivos[idx]

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

        tokens_pergunta = set(pergunta_lower.split())
        tokens_chunk = set(conteudo.lower().split())
        intersecao = tokens_pergunta.intersection(tokens_chunk)

        peso_embedding = 0.5
        peso_texto = 0.5
        similaridade = round(peso_embedding * similaridade_base + peso_texto * melhor_score, 4)

        if len(intersecao) >= 2:
            similaridade = min(similaridade + 0.05, 1.0)
        if melhor_score > 0.92:
            similaridade = min(similaridade + 0.03, 1.0)

        resultados.append({
            "nome": nome_arquivo,
            "trecho": trecho,
            "link": encontrar_link(nome_arquivo),
            "similaridade": similaridade
        })

    resultados.sort(key=lambda x: x["similaridade"], reverse=True)
    return resultados[:5]

def busca_literal_em_todos(pergunta, limite=0.4):
    resultados = []
    pergunta_normalizada = normalizar(pergunta)

    # Busca por número e ano no nome dos arquivos
    match = re.search(r'\b0*(\d{1,4})(?:[\s_/-]*(\d{2,4}))?\b', pergunta)
    padroes = []
    if match:
        numero, ano = match.groups()
        numero_formatado = f"{int(numero):03d}"
        if ano:
            padroes.extend([
                f"{numero_formatado}{ano}",
                f"{numero_formatado}_{ano}",
                f"{numero_formatado}-{ano}",
                f"{numero_formatado}/{ano}",
                f"{numero_formatado} {ano}"
            ])
        else:
            padroes.append(numero_formatado)
        padroes = [normalizar(p) for p in padroes]

        for nome, conteudo in zip(nome_arquivos, chunks):
            nome_normalizado = normalizar(nome)
            if any(p in nome_normalizado for p in padroes):
                trecho = textwrap.shorten(conteudo.replace("\n", " "), width=400, placeholder=" [...]")
                resultados.append({
                    "nome": nome,
                    "trecho": trecho,
                    "link": encontrar_link(nome),
                    "similaridade": 1.0
                })

        if resultados:
            return sorted(resultados, key=lambda x: x["nome"])

    # Fallback: comparação difusa
    for nome, conteudo in zip(nome_arquivos, chunks):
        nome_normalizado = normalizar(nome)
        ratio_nome = SequenceMatcher(None, pergunta_normalizada, nome_normalizado).ratio()
        conteudo_normalizado = normalizar(conteudo)
        ratio_conteudo = SequenceMatcher(None, conteudo_normalizado, pergunta_normalizada).ratio()

        score = max(ratio_nome, ratio_conteudo)
        if score >= limite:
            trecho = textwrap.shorten(conteudo.replace("\n", " "), width=400, placeholder=" [...]")
            resultados.append({
                "nome": nome,
                "trecho": trecho,
                "link": encontrar_link(nome),
                "similaridade": round(score, 2)
            })

    resultados.sort(key=lambda x: x["similaridade"], reverse=True)
    return resultados[:5]