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

modelo = SentenceTransformer('all-mpnet-base-v2')
def carregar_index():
    return faiss.read_index("index_faiss.idx")


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
    nome_base = re.sub(r'_chunk\d+\.txt$', '', nome_arquivo) + ".txt"
    chave_normalizada = normalizar(nome_base)
    
    for chave in mapeamento_links:
        chave_norm = normalizar(chave)
        
        if chave_norm == chave_normalizada:
            return mapeamento_links[chave]
        if chave_norm.startswith(chave_normalizada):
            return mapeamento_links[chave]
    
    print(f"[DEBUG] Não encontrou link para '{nome_arquivo}' (normalizado: {chave_normalizada})")
    return None

def busca_semantica(pergunta, top_k=3, limiar=0.75):
    index = carregar_index()
    embedding = modelo.encode([pergunta], convert_to_numpy=True, normalize_embeddings=True)
    distancias, indices = index.search(embedding, top_k)
    resultados = []

    for idx in indices[0]:
        if idx >= len(nomes):
            continue
        nome = nomes[idx]
        caminho = os.path.join("textos_extraidos", nome)
        if not os.path.exists(caminho):
            continue
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
        distancia = distancias[0][list(indices[0]).index(idx)]
        similaridade = round(1 - (distancia / 2), 2)

        if similaridade < limiar:
            continue

        resultados.append({
            "nome": nome,
            "trecho": trecho,
            "link": link,
            "similaridade": similaridade
        })

    # Ordenar os resultados do maior para o menor
    resultados.sort(key=lambda x: x['similaridade'], reverse=True)
    return resultados

def busca_literal_em_todos(pergunta, limite=0.4):
    import re
    resultados = []
    pergunta_normalizada = normalizar(pergunta)

    match = re.search(r'\b0*(\d{1,4})(?:[\s_/-]*(\d{2,4}))?\b', pergunta_normalizada)

    if match:
        numero, ano = match.groups()
        numero_formatado = f"{int(numero):03d}"

        padroes = [numero_formatado]
        if ano:
            padroes.append(f"{numero_formatado}{ano}")

        for nome in nomes:
            nome_normalizado = normalizar(nome)
            if any(p in nome_normalizado for p in padroes):
                caminho = os.path.join("textos_extraidos", nome)
                if not os.path.exists(caminho):
                    continue
                with open(caminho, "r", encoding="utf-8") as f:
                    conteudo = f.read()
                trecho = textwrap.shorten(conteudo.replace("\n", " "), width=400, placeholder=" [...]")
                link = encontrar_link(nome)
                resultados.append({
                    "nome": nome,
                    "trecho": trecho,
                    "link": link,
                    "similaridade": 1.0
                })

        if resultados:
            return sorted(resultados, key=lambda x: x["nome"])

    # Fallback: busca difusa
    for nome in nomes:
        nome_normalizado = normalizar(nome)
        ratio_nome = SequenceMatcher(None, pergunta_normalizada, nome_normalizado).ratio()

        caminho = os.path.join("textos_extraidos", nome)
        if not os.path.exists(caminho):
            continue
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
    return resultados[:5]