import sys
from extrator import processar_pdfs
from gerar_embeddings import gerar_embeddings
from buscar import busca_semantica, busca_literal_em_todos
from sentence_transformers import SentenceTransformer
import subprocess

PDFS_IGNORAR = [
    "Documento_de_área_2019.pdf",
    "Documento_de_área_2017.pdf",
    "Documento_de_área_2013_-_Regras_do_CA-CC_para_Avaliação_de_Mestrados_E_Qualis_Periódicos_(WebQualis).pdf",
    "Documento_de_área_2013_-_Regras_do_CA-CC_para_Avaliação_de_Mestrados.pdf"
]

MODELOS = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    'all-mpnet-base-v2'
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

    print("\n2. Gerando embeddings e construindo índices FAISS...")
    gerar_embeddings()

def loop_busca():
    print("\n== BUSCA INTERATIVA ==")
    print("Escolha o modelo:")
    for idx, m in enumerate(MODELOS):
        print(f"{idx+1}. {m}")

    opcao = input("Digite o número do modelo: ").strip()
    try:
        modelo_nome = MODELOS[int(opcao)-1]
    except:
        print("Opção inválida. Saindo.")
        return

    modelo = SentenceTransformer(modelo_nome)

    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if not pergunta:
            print("Por favor, digite uma pergunta válida.")
            continue
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando a busca.")
            break

        resultados_sem = busca_semantica(pergunta, modelo, modelo_nome)

        if not resultados_sem:
            print("Nenhum resultado semântico. Fazendo busca literal...")
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

def executar_teste():
    print("\n== EXECUTANDO BENCHMARK E GERANDO GRÁFICOS ==")
    print("Executando benchmark_busca.py ...")
    subprocess.run([sys.executable, "benchmark/benchmark_busca.py"])
    print("Executando plot_benchmark.py ...")
    subprocess.run([sys.executable, "benchmark/plot_benchmark.py"])
    print("✅ Benchmark e gráficos concluídos.")

if __name__ == "__main__":
    print("== Projeto: Busca Semântica de Legislação PPGCAP ==")
    print("Escolha a opção:")
    print("1. Executar busca interativa")
    print("2. Executar benchmark e gerar gráficos")
    print("3. Executar pipeline completa (extração + embeddings)")

    escolha = input("Digite 1, 2 ou 3: ").strip()

    if escolha == "1":
        loop_busca()
    elif escolha == "2":
        executar_teste()
    elif escolha == "3":
        executar_pipeline()
    else:
        print("Opção inválida. Encerrando.")