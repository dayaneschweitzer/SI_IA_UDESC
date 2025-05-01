import sys
from extrator import processar_pdfs
from gerar_embeddings import gerar_embeddings
from buscar import busca_semantica

# Ignorar PDFs específicos (os mesmos da sua configuração anterior)
PDFS_IGNORAR = [
    "Documento_de_área_2019.pdf",
    "Documento_de_área_2017.pdf",
    "Documento_de_área_2013_-_Regras_do_CA-CC_para_Avaliação_de_Mestrados_E_Qualis_Periódicos_(WebQualis).pdf",
    "Documento_de_área_2013_-_Regras_do_CA-CC_para_Avaliação_de_Mestrados.pdf"
]

def executar_pipeline():
    print("1. Extraindo e dividindo documentos em chunks...")
    erros = processar_pdfs(pasta_pdfs="portarias", ignorar=PDFS_IGNORAR)
    if erros:
        print("\nAlguns arquivos não puderam ser processados:")
        for e in erros:
            print(f"- {e}")
    else:
        print("Todos os PDFs foram processados com sucesso.")

    print("\n2. Gerando embeddings e construindo índice FAISS...")
    gerar_embeddings()

def loop_interativo():
    print("\n3. Busca Interativa por Perguntas")
    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando a busca.")
            break

        resultados = busca_semantica(pergunta, top_k=3)
        if not resultados:
            print("Nenhum resultado encontrado.")
        else:
            for i, r in enumerate(resultados, 1):
                print(f"\n{i}. Documento: {r['nome']}")
                print(f"Trecho: {r['trecho']}")
                print(f"Similaridade: {r['similaridade']}")
                if r['link']:
                    print(f"Link: {r['link']}")
                print("-" * 60)

if __name__ == "__main__":
    print("== Projeto: Busca Semântica de Legislação PPGCAP ==")
    executar_pipeline()
    loop_interativo()
