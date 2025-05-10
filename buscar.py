from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import json
import numpy as np
import textwrap
from unidecode import unidecode
from difflib import SequenceMatcher

modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
index = faiss.read_index("index_faiss.idx")

with open("nomes_textos.pkl", "rb") as f:
    nomes = pickle.load(f)

with open("mapeamento_links.json", "r", encoding="utf-8") as f:
    mapeamento_links = json.load(f)

def normalizar(texto):
    texto = unidecode(texto.lower())
    texto = texto.replace("_", " ").replace("-", " ")
    texto = re.sub(r'[\W_]+', ' ', texto)
    return texto.strip()

def encontrar_link(nome_arquivo):
    nome_base = re.sub(r'_chunk\d+\.txt$', '', nome_arquivo)
    chave_normalizada = normalizar(nome_base)
    for chave in mapeamento_links:
        if normalizar(chave).startswith(chave_normalizada):
            return mapeamento_links[chave]
    return None

def busca_semantica(pergunta, top_k=3):
    embedding = modelo.encode([pergunta], convert_to_numpy=True)
    distancias, indices = index.search(embedding, top_k)
    resultados = []
    for idx in indices[0]:
        if idx >= len(nomes): continue
        nome = nomes[idx]
        caminho = os.path.join("textos_extraidos", nome)
        if not os.path.exists(caminho): continue
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read()
        pergunta_lower = pergunta.lower()
        trecho = ""
        for linha in conteudo.splitlines():
            if any(p in linha.lower() for p in pergunta_lower.split()):
                trecho = linha.strip()
                break
        if not trecho:
            trecho = textwrap.shorten(conteudo.replace("\n", " "), width=300, placeholder=" [...]")
        link = encontrar_link(nome)
        similaridade = round(1 / (1 + distancias[0][list(indices[0]).index(idx)]), 2)
        resultados.append({
            "nome": nome,
            "trecho": trecho,
            "link": link,
            "similaridade": similaridade
        })
    return resultados

def busca_literal_em_todos(pergunta, limite=0.4):
    resultados = []
    pergunta_normalizada = normalizar(pergunta)

    for nome in nomes:
        nome_normalizado = normalizar(nome)
        ratio_nome = SequenceMatcher(None, pergunta_normalizada, nome_normalizado).ratio()

        caminho = os.path.join("textos_extraidos", nome)
        if not os.path.exists(caminho): continue
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read()
            conteudo_normalizado = unidecode(conteudo.lower())
            ratio_conteudo = SequenceMatcher(None, conteudo_normalizado, pergunta_normalizada).ratio()

        score = max(ratio_nome, ratio_conteudo)
        if score >= limite:
            trecho = textwrap.shorten(conteudo.replace("\n", " "), width=400, placeholder=" [...]")
            link = encontrar_link(nome)
            resultados.append({
                "nome": nome,
                "trecho": trecho,
                "link": link,
                "similaridade": round(score, 2)
            })

    resultados.sort(key=lambda x: x["similaridade"], reverse=True)
    return resultados