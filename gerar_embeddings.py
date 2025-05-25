from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from sklearn.decomposition import PCA

MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
]

N_COMPONENTS_PCA = 256

DIRETORIO_TEXTOS = "textos_extraidos"

for modelo_nome in MODELOS:
    print(f"\n🔹 Processando modelo: {modelo_nome}")
    modelo = SentenceTransformer(modelo_nome)

    textos = []
    nomes = []

    for nome_arquivo in os.listdir(DIRETORIO_TEXTOS):
        if nome_arquivo.endswith(".txt"):
            caminho = os.path.join(DIRETORIO_TEXTOS, nome_arquivo)
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                nome_no_texto = nome_arquivo.replace("_", " ").replace(".txt", "")
                texto_com_nome = nome_no_texto + " " + conteudo
                textos.append(texto_com_nome)
            nomes.append(nome_arquivo)

    print(f"🔸 Gerando embeddings para {len(nomes)} textos...")
    embeddings = modelo.encode(textos, convert_to_numpy=True, normalize_embeddings=True)

    print(f"🔸 Aplicando PCA para redução a {N_COMPONENTS_PCA} dimensões...")
    pca = PCA(n_components=N_COMPONENTS_PCA)
    embeddings_reduzidos = pca.fit_transform(embeddings)

    dimension = embeddings_reduzidos.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_reduzidos)

    # Salvando arquivos com nome do modelo
    modelo_safe = modelo_nome.replace("/", "_")

    with open(f'embeddings_{modelo_safe}.pkl', 'wb') as f:
        pickle.dump(embeddings_reduzidos, f)

    with open(f'pca_model_{modelo_safe}.pkl', 'wb') as f:
        pickle.dump(pca, f)

    faiss.write_index(index, f'index_faiss_{modelo_safe}.idx')

    with open(f'nomes_textos_{modelo_safe}.pkl', 'wb') as f:
        pickle.dump(nomes, f)

    print(f" Arquivos salvos para modelo {modelo_safe}:")
    print(f"    embeddings_{modelo_safe}.pkl")
    print(f"    pca_model_{modelo_safe}.pkl")
    print(f"    index_faiss_{modelo_safe}.idx")
    print(f"    nomes_textos_{modelo_safe}.pkl")
