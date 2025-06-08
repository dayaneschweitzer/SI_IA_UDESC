import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import textwrap

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

textos_folder = "textos_extraidos"
vetores_folder = "vetores"
os.makedirs(vetores_folder, exist_ok=True)

modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

embeddings = []
metadados = []

for filename in os.listdir(textos_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(textos_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
            chunks = chunk_text(texto, chunk_size=1000, overlap=200)
            
            for i, chunk in enumerate(chunks):
                embedding = modelo.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
                metadados.append({
                    "arquivo": filename,
                    "chunk_id": i,
                    "texto": chunk
                })

embeddings = np.array(embeddings)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(vetores_folder, "faiss_index.index"))
with open(os.path.join(vetores_folder, "metadados.pkl"), 'wb') as f:
    pickle.dump(metadados, f)

print(f"Gerados {len(embeddings)} chunks e índice FAISS salvo.")