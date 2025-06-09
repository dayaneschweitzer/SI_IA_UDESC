import os
import pickle
import faiss
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer

# Função para dividir texto em chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Pastas
textos_folder = "textos_extraidos"
vetores_folder = "vetores"
pdf_base_path = "PDFs_Udesc"

os.makedirs(vetores_folder, exist_ok=True)

# Carregar JSON com informações dos documentos
with open(os.path.join(pdf_base_path, "resolucoes_ppgcap.json"), 'r', encoding='utf-8') as f:
    resolucoes_info = json.load(f)

# Criar um mapa de prefixos → arquivo real
prefixo_para_arquivo_real = {}

for item in resolucoes_info:
    arquivo_json = item["arquivo"]
    prefixo = re.sub(r'_V\d+.*\.pdf$', '', arquivo_json)  # remover _Vxxx... se existir
    prefixo = re.sub(r'[\-_]', '', prefixo).lower()       # normalizar prefixo
    prefixo_para_arquivo_real[prefixo] = arquivo_json

# Modelo de embeddings
modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

# Vetores e metadados
embeddings = []
metadados = []

# Processar cada texto
for filename in os.listdir(textos_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(textos_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
            chunks = chunk_text(texto, chunk_size=1000, overlap=200)

            # Extrair prefixo do chunk
            prefixo_filename = re.sub(r'_chunk_\d+\.txt$', '', filename)
            prefixo_filename = re.sub(r'[\-_]', '', prefixo_filename).lower()

            # Buscar no mapa
            arquivo_real_pdf = prefixo_para_arquivo_real.get(prefixo_filename, None)

            if not arquivo_real_pdf:
                print(f"[WARNING] Não achei correspondência para '{prefixo_filename}' — este chunk será ignorado.")
                continue  # Pular chunks sem correspondência

            # Buscar informações do documento
            info_doc = next((item for item in resolucoes_info if item["arquivo"] == arquivo_real_pdf), {})
            titulo = info_doc.get("titulo", arquivo_real_pdf)
            link = info_doc.get("link", None)  # None se não tiver no JSON

            # Processar chunks
            for i, chunk in enumerate(chunks):
                embedding = modelo.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
                metadados.append({
                    "arquivo": arquivo_real_pdf,  # salvar com nome correto do JSON
                    "chunk_id": i,
                    "titulo": titulo,
                    "link": link,
                    "texto": chunk
                })

# Criar índice FAISS
embeddings = np.array(embeddings)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Salvar índice e metadados
faiss.write_index(index, os.path.join(vetores_folder, "faiss_index.index"))
with open(os.path.join(vetores_folder, "metadados.pkl"), 'wb') as f:
    pickle.dump(metadados, f)

print(f"\n✅ Gerados {len(embeddings)} chunks e índice FAISS salvo.")
print(f"✅ metadados.pkl com 'arquivo' igual ao JSON.")
