import os
import time
import pickle
import json
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

with open(os.path.join(pdf_base_path, "resolucoes_ppgcap.json"), 'r', encoding='utf-8') as f:
    resolucoes_info = json.load(f)

mapa_arquivos = {
    item["arquivo"]: {
        "titulo": item["titulo"],
        "link": item["link"]
    }
    for item in resolucoes_info
}

modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

llm = OllamaLLM(model="mistral")

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]
    pergunta = ""
    resposta_final = ""
    tempo_resposta = 0
    resultados_sem = []

    if not chat_history:
        saudacao = "Olá! Sou o assistente do PPGCAP. Como posso ajudar você com as legislações? 📚"
        chat_history.append({
            "pergunta": "",
            "resposta": saudacao
        })
        session["chat_history"] = chat_history

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()

        if pergunta.lower() in ["oi", "olá", "bom dia", "boa tarde", "boa noite"]:
            resposta_final = "Olá! Como posso te ajudar? Você pode me perguntar sobre qualquer legislação do PPGCAP. 📚"
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=[],
                                   tempo=tempo_resposta)

        if pergunta.lower() == "encontrei meu arquivo":
            resposta_final = "Excelente! Se quiser fazer mais buscas, estou por aqui. 😊"
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=[],
                                   tempo=tempo_resposta)

        if pergunta:
            inicio = time.time()
            try:
                embedding_pergunta = modelo.encode(pergunta, convert_to_numpy=True)
                k = 2
                D, I = faiss_index.search(np.array([embedding_pergunta]), k)

                documentos_relevantes = {}

                for distancia, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    doc_meta = metadados[idx]

                    arquivo_original = doc_meta.get("arquivo", "")
                    arquivo_pdf_nome = arquivo_original.split("_chunk_")[0].replace(".txt", ".pdf")

                    info_json = mapa_arquivos.get(arquivo_pdf_nome, None)

                    if info_json:
                        titulo = info_json["titulo"]
                        link = info_json["link"]
                    else:
                        titulo = doc_meta.get("titulo", arquivo_original)
                        link = doc_meta.get("link", "#")

                    trecho_resumido = doc_meta['texto'][:150].replace("\n", " ").strip() + "..."

                    if titulo not in documentos_relevantes:
                        documentos_relevantes[titulo] = {
                            "link": link,
                            "trecho": trecho_resumido
                        }

                conteudo_docs_texto = "Aqui estão os documentos relevantes:\n\n"
                for titulo, doc_info in documentos_relevantes.items():
                    conteudo_docs_texto += f"\n--- Documento: {titulo} ---\n"
                    conteudo_docs_texto += f"Link: {doc_info['link']}\n"
                    conteudo_docs_texto += f"Trecho: {doc_info['trecho']}\n"

                resultados_sem = []
                for titulo, doc_info in documentos_relevantes.items():
                    resultados_sem.append({
                        "nome": titulo,
                        "trecho": doc_info['trecho'],
                        "link": doc_info['link']
                    })

                prompt = f"""
Você é um assistente especializado em legislação do PPGCAP.

Seu papel é ajudar o usuário a encontrar informações nas resoluções oficiais do programa.

Você deve responder de forma clara e amigável.

IMPORTANTE: você só pode responder com base nos documentos listados abaixo. 
Se a informação não estiver presente, diga: "Não encontrei essa informação nos documentos."

{conteudo_docs_texto}

Histórico da conversa:
"""

                historico_recente = chat_history[-1:]

                for turn in historico_recente:
                    if turn["pergunta"]:
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

if __name__ == "__main__":
    app.run(debug=True)
