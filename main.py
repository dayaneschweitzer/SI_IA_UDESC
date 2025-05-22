import sys
from extrator import processar_pdfs
from gerar_embeddings import gerar_embeddings
from buscar import busca_semantica, busca_literal_em_todos

# Ignorar PDFs específicos
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
        print("\nArquivos com erro de leitura:")
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
        if not pergunta:
            print("Por favor, digite uma pergunta válida.")
            continue
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando a busca.")
            break

        resultados_sem = busca_semantica(pergunta)

        if not resultados_sem:
            resultados_lit = busca_literal_em_todos(pergunta)
            for r in resultados_lit:
                print(f"\nDocumento: {r['nome']}")
                print(f"Trecho: {r['trecho']}")
                print(f"Link: {r['link']}")
                print(f"Similaridade: {r['similaridade']}")
                print("-" * 60)
        else:
            for r in resultados_sem:
                print(f"\nDocumento: {r['nome']}")
                print(f"Trecho: {r['trecho']}")
                print(f"Link: {r['link']}")
                print(f"Similaridade: {r['similaridade']}")
                print("-" * 60)

if __name__ == "__main__":
    print("== Projeto: Busca Semântica de Legislação PPGCAP ==")
    executar_pipeline()
    loop_interativo()
