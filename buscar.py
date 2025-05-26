from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import json
import time
import numpy as np
import textwrap
from unidecode import unidecode
from difflib import SequenceMatcher

# Função de normalização
def normalizar(texto):
    texto = unidecode(texto.lower())
    texto = texto.replace("_", " ").replace("-", " ")
    texto = re.sub(r'[\W_]+', ' ', texto)
    return texto.strip()

# Encontrar link
with open("mapeamento_links.json", "r", encoding="utf-8") as f:
    mapeamento_links = json.load(f)

def encontrar_link(nome_arquivo):
    nome_base = re.sub(r'_chunk\d+\.txt$', '', nome_arquivo) + ".txt"
    chave_normalizada = normalizar(nome_base)
    
    for chave in mapeamento_links:
        chave_norm = normalizar(chave)
        if chave_norm == chave_normalizada or chave_norm.startswith(chave_normalizada):
            return mapeamento_links[chave]
    
    print(f"[DEBUG] Não encontrou link para '{nome_arquivo}' (normalizado: {chave_normalizada})")
    return None

# Função para carregar os recursos por modelo e PCA
def carregar_recursos(modelo_nome, n_pca):
    modelo_safe = modelo_nome.replace("/", "_")
    suf = f"{modelo_safe}_pca{n_pca}"
    
    index = faiss.read_index(f'index_faiss_{suf}.idx')
    
    with open(f'pca_model_{suf}.pkl', 'rb') as f:
        pca = pickle.load(f)
    
    with open(f'nomes_textos_{suf}.pkl', 'rb') as f:
        nomes = pickle.load(f)
    
    return index, pca, nomes

# Busca semântica com PCA
def busca_semantica(pergunta, modelo, modelo_nome, n_pca, top_k=3, limiar=0.75):
    try:
        index, pca, nomes = carregar_recursos(modelo_nome, n_pca)
    except FileNotFoundError as e:
        print(f"Arquivo de recurso não encontrado para {modelo_nome} com PCA {n_pca}: {e}")
        return [], 0.0

    start = time.time()

    embedding = modelo.encode([pergunta], convert_to_numpy=True, normalize_embeddings=True)
    embedding_reduzido = pca.transform(embedding)
    distancias, indices = index.search(embedding_reduzido, top_k)

    end = time.time()
    tempo = end - start  # tempo de consulta

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

    resultados.sort(key=lambda x: x['similaridade'], reverse=True)
    return resultados, tempo

# Busca literal com normalização e difusa
def busca_literal_em_todos(pergunta, modelo_nome, n_pca, limite=0.4):
    modelo_safe = modelo_nome.replace("/", "_")
    suf = f"{modelo_safe}_pca{n_pca}"
    
    with open(f'nomes_textos_{suf}.pkl', 'rb') as f:
        nomes = pickle.load(f)
    
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