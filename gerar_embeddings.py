from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

def gerar_embeddings(diretorio_textos="textos_extraidos", nome_index="index_faiss.idx", nome_lista="nomes_textos.pkl"):
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    textos = []
    nomes = []

    for nome_arquivo in os.listdir(diretorio_textos):
        if nome_arquivo.endswith(".txt"):
            with open(os.path.join(diretorio_textos, nome_arquivo), "r", encoding="utf-8") as f:
                textos.append(f.read())
            nomes.append(nome_arquivo)

    embeddings = modelo.encode(textos, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, nome_index)
    with open(nome_lista, "wb") as f:
        pickle.dump(nomes, f)

    print(f"{len(nomes)} embeddings salvos no índice FAISS.")
