import os
import time
import pickle
import faiss
import numpy as np
from flask import Flask, request, render_template, session
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

pdf_base_path = os.path.abspath("PDFs_Udesc")
vetores_folder = os.path.abspath("vetores")

app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta_para_sessao" 

faiss_index = faiss.read_index(os.path.join(vetores_folder, "faiss_index.index"))

with open(os.path.join(vetores_folder, "metadados.pkl"), 'rb') as f:
    metadados = pickle.load(f)

modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

llm = OllamaLLM(model="llama3:8b")

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]
    pergunta = ""
    resposta_final = ""
    tempo_resposta = 0
    resultados_sem = []

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()
        if pergunta:
            inicio = time.time()
            try:
                embedding_pergunta = modelo.encode(pergunta, convert_to_numpy=True)
                k = 5
                D, I = faiss_index.search(np.array([embedding_pergunta]), k)

                conteudo_docs = ""
                resultados_sem = []

                for distancia, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    doc_meta = metadados[idx]

                    chunk_texto = doc_meta['texto'][:1000] + "..."  # limitar o chunk

                    conteudo_docs += f"\n--- Documento: {doc_meta['arquivo']} (chunk {doc_meta.get('chunk_id', 0)}) ---\n"
                    conteudo_docs += chunk_texto + "\n"

                    resultados_sem.append({
                        "nome": doc_meta.get('titulo', doc_meta['arquivo']),
                        "trecho": chunk_texto,
                        "link": doc_meta.get('link', f"/documento/{doc_meta['arquivo']}")
                    })

                prompt = """
Você é um assistente especializado em legislação do PPGCAP.

Seu objetivo é responder perguntas do usuário **somente com base nos documentos fornecidos abaixo**.

Se a informação não estiver nos documentos, diga: "Não encontrei essa informação nos documentos."

### Documentos relevantes:
""" + conteudo_docs + """

### Histórico da conversa:
"""

                for turn in chat_history:
                    prompt += f"\nUsuário: {turn['pergunta']}\nAssistente: {turn['resposta']}"

                prompt += f"\nUsuário: {pergunta}\nAssistente: "

                resposta_final = llm.invoke(prompt)

                chat_history.append({
                    "pergunta": pergunta,
                    "resposta": resposta_final
                })
                session["chat_history"] = chat_history

            except Exception as e:
                resposta_final = f"Erro ao buscar ou gerar resposta: {str(e)}"

            tempo_resposta = round(time.time() - inicio, 3)

    return render_template("index.html",
                           chat_history=chat_history,
                           pergunta_atual=pergunta,
                           resposta_atual=resposta_final,
                           resultados_sem=resultados_sem,
                           tempo=tempo_resposta)

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

@app.route("/documento/<path:nome>")
def documento(nome):
    return send_from_directory(pdf_base_path, nome)

if __name__ == "__main__":
    app.run(debug=True)