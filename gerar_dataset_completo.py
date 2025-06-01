import os
import pickle
from sentence_transformers import SentenceTransformer

def gerar_dataset_completo(diretorio_textos="textos_extraidos", saida="dados_completos.pkl"):
    modelo_nome = "paraphrase-multilingual-MiniLM-L12-v2"
    modelo = SentenceTransformer(modelo_nome)

    nomes_arquivos = []
    chunks = []

    for nome in os.listdir(diretorio_textos):
        if nome.endswith(".txt"):
            caminho = os.path.join(diretorio_textos, nome)
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                nomes_arquivos.append(nome)
                chunks.append(conteudo)

    embeddings = modelo.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dados = {
        "nomes_arquivos": nomes_arquivos,
        "chunks": chunks,
        "embeddings": embeddings
    }

    with open(saida, "wb") as f:
        pickle.dump(dados, f)

    print(f"Arquivo salvo com {len(chunks)} chunks em {saida}")

if __name__ == "__main__":
    gerar_dataset_completo()
