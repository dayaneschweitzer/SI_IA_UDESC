import os
import re
import pickle
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

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
pdf_base_path = "PDFs_Udesc"

os.makedirs(vetores_folder, exist_ok=True)

with open(os.path.join(pdf_base_path, "resolucoes_ppgcap.json"), 'r', encoding='utf-8') as f:
    resolucoes_info = json.load(f)

informacoes_documentos = {}
for item in resolucoes_info:
    informacoes_documentos[item["arquivo"]] = {
        "titulo": item["titulo"],
        "link": item["link"]
    }

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

            nome_base_pdf = re.sub(r"_chunk_\d+\.txt$", ".pdf", filename)

            info_doc = informacoes_documentos.get(nome_base_pdf, {})
            titulo = info_doc.get("titulo", nome_base_pdf)
            link = info_doc.get("link", f"/documento/{filename}")

            for i, chunk in enumerate(chunks):
                embedding = modelo.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
                metadados.append({
                    "arquivo": filename,
                    "chunk_id": i,
                    "titulo": titulo,
                    "link": link,
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