import sys
from buscar import buscar_resposta
from sentence_transformers import SentenceTransformer

MODELO_NOME = "all-mpnet-base-v2"
modelo = SentenceTransformer(MODELO_NOME)

def loop_interativo(modelo):
    print("\n== Modo Interativo de Busca ==")
    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if not pergunta:
            print("Por favor, digite uma pergunta válida.")
            continue
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando a busca.")
            break

        resultados, tempo = buscar_resposta(pergunta)
        print(f"\nTempo de resposta: {tempo} segundos")
        if not resultados:
            print("Nenhum resultado encontrado.")
        else:
            for r in resultados:
                print(f"\nDocumento: {r['arquivo']}")
                print(f"Trecho: {r['trecho']}")
                print(f"Similaridade: {r['similaridade']}")
                print("-" * 60)

if __name__ == "__main__":
    print("== Projeto: Busca Semântica de Legislação PPGCAP ==")
    loop_interativo(modelo)
