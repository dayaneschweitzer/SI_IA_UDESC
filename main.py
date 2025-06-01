# main.py

from buscar import busca_semantica, busca_literal_em_todos
from sentence_transformers import SentenceTransformer

MODELO_NOME = "all-mpnet-base-v2"
modelo = SentenceTransformer(MODELO_NOME)

def loop_interativo():
    print("=== Busca Interativa PPGCAP ===")
    while True:
        pergunta = input("\nPergunta (ou 'sair'): ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        resultados = busca_semantica(pergunta, modelo, MODELO_NOME)
        if resultados:
            print("\n--- Resultados Semânticos ---")
            for r in resultados:
                print(f"{r['nome']}\n{r['trecho']}\n{r['link']}\nSimilaridade: {r['similaridade']}\n")
        else:
            resultados_lit = busca_literal_em_todos(pergunta)
            if resultados_lit:
                print("\n--- Resultados Literais ---")
                for r in resultados_lit:
                    print(f"{r['nome']}\n{r['trecho']}\n{r['link']}\nSimilaridade: {r['similaridade']}\n")
            else:
                print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    loop_interativo()