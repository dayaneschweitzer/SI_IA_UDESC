from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

def gerar_embeddings(diretorio_textos="textos_extraidos", nome_lista="nomes_textos.pkl"):
    modelo_nome = 'paraphrase-multilingual-MiniLM-L12-v2'
    modelo = SentenceTransformer(modelo_nome)
    modelo_nome_safe = modelo_nome.replace("/", "_")
    nome_index = f"index_faiss_{modelo_nome_safe}.idx"

    textos = []
    nomes = []

    for nome_arquivo in os.listdir(diretorio_textos):
        if nome_arquivo.endswith(".txt"):
            caminho = os.path.join(diretorio_textos, nome_arquivo)
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                nome_no_texto = nome_arquivo.replace("_", " ").replace(".txt", "")
                texto_com_nome = nome_no_texto + " " + conteudo
                textos.append(texto_com_nome)
            nomes.append(nome_arquivo)

    embeddings = modelo.encode(textos, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, nome_index)
    with open(nome_lista, "wb") as f:
        pickle.dump(nomes, f)

    print(f"{len(nomes)} embeddings salvos no índice FAISS: {nome_index}")
