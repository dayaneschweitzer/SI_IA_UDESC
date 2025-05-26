from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from sklearn.decomposition import PCA

def gerar_embeddings():
    MODELOS = [
        'paraphrase-multilingual-MiniLM-L12-v2',
        'all-mpnet-base-v2'
    ]

    N_COMPONENTS_PCA = [256, 512, 768, 1024]
    DIRETORIO_TEXTOS = "textos_extraidos"

    for modelo_nome in MODELOS:
        print(f"\n Processando modelo: {modelo_nome}")
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

        print(f"Gerando embeddings para {len(nomes)} textos...")
        embeddings = modelo.encode(textos, convert_to_numpy=True, normalize_embeddings=True)

        # Definir o máximo de componentes permitidos pelo PCA
        n_max = min(embeddings.shape[0], embeddings.shape[1])

        # Filtra apenas as dimensões válidas
        pca_dimensoes = [n for n in N_COMPONENTS_PCA if n <= n_max]

        modelo_safe = modelo_nome.replace("/", "_")

        for n_pca in pca_dimensoes:
            print(f"Aplicando PCA para redução a {n_pca} dimensões...")
            pca = PCA(n_components=n_pca)
            embeddings_reduzidos = pca.fit_transform(embeddings)

            dimension = embeddings_reduzidos.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_reduzidos)

            suf = f"{modelo_safe}_pca{n_pca}"

            with open(f'embeddings_{suf}.pkl', 'wb') as f:
                pickle.dump(embeddings_reduzidos, f)

            with open(f'pca_model_{suf}.pkl', 'wb') as f:
                pickle.dump(pca, f)

            faiss.write_index(index, f'index_faiss_{suf}.idx')

            with open(f'nomes_textos_{suf}.pkl', 'wb') as f:
                pickle.dump(nomes, f)

            print(f"Arquivos salvos para {suf}:")
            print(f"    embeddings_{suf}.pkl")
            print(f"    pca_model_{suf}.pkl")
            print(f"    index_faiss_{suf}.idx")
            print(f"    nomes_textos_{suf}.pkl")

        # Aviso se nenhuma dimensão foi possível
        if not pca_dimensoes:
            print(f"Nenhuma dimensão de PCA aplicável para modelo {modelo_safe}. Máximo permitido: {n_max}")

# Execução direta
if __name__ == "__main__":
    gerar_embeddings()
