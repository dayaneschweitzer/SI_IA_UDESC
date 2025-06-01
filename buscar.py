import os
import pickle
import time
import textwrap
import re
import unicodedata
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_DISPONIVEL = True
except ImportError:
    RAPIDFUZZ_DISPONIVEL = False

# Carrega dados
data_path = os.path.join(os.path.dirname(__file__), "dados_completos.pkl")
with open(data_path, "rb") as f:
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

def calcula_score(seq1, seq2, metric="sequence"):
    if metric == "rapidfuzz" and RAPIDFUZZ_DISPONIVEL:
        return fuzz.token_sort_ratio(seq1, seq2) / 100
    return SequenceMatcher(None, seq1, seq2).ratio()

def busca_semantica(pergunta, modelo, modelo_nome="all-mpnet-base-v2", top_k=5, 
                    peso_embedding=0.5, peso_texto=0.5, metric="sequence", index=None):

    pergunta_lower = pergunta.lower()
    pergunta_embedding = modelo.encode([pergunta], convert_to_numpy=True)
    
    # Usa índice existente ou cria dinamicamente
    if index is None:
        embedding_dim = pergunta_embedding.shape[1]
        index = NearestNeighbors(n_neighbors=top_k, metric="cosine")
        index.fit(embeddings)

    distancias, indices = index.kneighbors(pergunta_embedding, n_neighbors=top_k)
    resultados = []
    inicio = time.time()

    for idx in indices[0]:
        conteudo = chunks[idx]
        nome_arquivo = nome_arquivos[idx]

        melhor_linha = ""
        melhor_score = 0.0
        for linha in conteudo.splitlines():
            score = calcula_score(pergunta_lower, linha.lower(), metric)
            if score > melhor_score:
                melhor_score = score
                melhor_linha = linha

        trecho = melhor_linha.strip() if melhor_linha else textwrap.shorten(conteudo.replace("\n", " "), width=300)
        distancia = distancias[0][list(indices[0]).index(idx)]
        similaridade_base = round(1 - (distancia / 2), 4)

        tokens_pergunta = set(pergunta_lower.split())
        tokens_chunk = set(conteudo.lower().split())
        intersecao = tokens_pergunta.intersection(tokens_chunk)

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
    tempo = round(time.time() - inicio, 3)
    return resultados[:top_k], tempo

def busca_literal_em_todos(pergunta, limite=0.4):
    resultados = []
    pergunta_normalizada = normalizar(pergunta)

    match = re.search(r'\b0*(\d{1,4})(?:[\s_/-]*(\d{2,4}))?\b', pergunta_normalizada)
    if match:
        numero, ano = match.groups()
        numero_formatado = f"{int(numero):03d}"
        padroes = [numero_formatado]
        if ano:
            padroes.append(f"{numero_formatado}{ano}")

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