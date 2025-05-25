import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]

with open("nomes_textos.pkl", "rb") as f:
    nomes = pickle.load(f)

textos = []
for nome in nomes:
    caminho = os.path.join("textos_extraidos", nome)
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            textos.append(f.read())
    else:
        textos.append("")

for modelo_nome in MODELOS:
    print(f"Gerando index FAISS para modelo: {modelo_nome}")
    modelo = SentenceTransformer(modelo_nome)
    embeddings = modelo.encode(textos, convert_to_numpy=True, normalize_embeddings=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    faiss.write_index(index, f"index_faiss_{modelo_nome}.idx")
    print(f"Index salvo como index_faiss_{modelo_nome}.idx")
