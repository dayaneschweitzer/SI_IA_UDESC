from flask import Flask, request, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
from buscar import busca_semantica, busca_literal_em_todos
import time

app = Flask(__name__)

# Modelo fixo
MODELO_NOME = "all-mpnet-base-v2"
modelo = SentenceTransformer(MODELO_NOME)

@app.route("/", methods=["GET", "POST"])
def index():
    resultados_sem = []
    resultados_lit = []
    pergunta = ""
    tempo_resposta = 0

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()
        if pergunta:
            inicio = time.time()
            try:
                resultados_sem = busca_semantica(pergunta, modelo, MODELO_NOME)
                resultados_lit = busca_literal_em_todos(pergunta)

                # Se a melhor semântica for fraca, desconsidera
                if not resultados_sem or (resultados_sem[0]["similaridade"] < 0.45):
                    resultados_sem = []
            except Exception as e:
                resultados_sem = [{
                    "nome": "Erro ao buscar",
                    "trecho": str(e),
                    "link": "#",
                    "similaridade": 0.0
                }]
            tempo_resposta = round(time.time() - inicio, 3)

    return render_template("index.html", pergunta=pergunta,
                           resultados_sem=resultados_sem,
                           resultados_lit=resultados_lit if not resultados_sem else [],
                           tempo=tempo_resposta)

# ✅ Rota para abrir documento diretamente
@app.route("/documento/<path:nome>")
def documento(nome):
    return send_from_directory("textos_extraidos", nome)

if __name__ == "__main__":
    app.run(debug=True)
