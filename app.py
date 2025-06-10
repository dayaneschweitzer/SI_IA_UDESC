import os
import time
import pickle
import json
import faiss
import numpy as np
import unicodedata
from flask import Flask, request, render_template, session
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

def normalizar_nome(nome):
    nome = nome.lower()
    nome = unicodedata.normalize('NFD', nome).encode('ascii', 'ignore').decode('utf-8')
    nome = nome.replace('ç', 'c')
    nome = nome.replace('_', '').replace('-', '').replace(' ', '')
    return nome

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
    normalizar_nome(item["arquivo"]): {
        "titulo": item["titulo"],
        "link": item["link"]
    }
    for item in resolucoes_info
}

modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

MODEL_NAME = "mistral"
llm = OllamaLLM(model=MODEL_NAME)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "resultados_sem" not in session:
        session["resultados_sem"] = []
    if "mostrar_3_arquivos" not in session:
        session["mostrar_3_arquivos"] = False

    chat_history = session["chat_history"]
    resultados_sem = session["resultados_sem"]
    mostrar_3_arquivos = session["mostrar_3_arquivos"]

    pergunta = ""
    resposta_final = ""
    tempo_resposta = 0
    exibir_botoes = False

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()

        if pergunta.lower() in ["oi", "olá", "bom dia", "boa tarde", "boa noite"]:
            resposta_final = "Olá! Eu sou o assistente do PPGCAP. Como posso ajudar você com as legislações? 📚"
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
                                   tempo=tempo_resposta,
                                   exibir_botoes=False,
                                   mostrar_3_arquivos=False)

        if pergunta.lower() == "não, já encontrei meu documento.":
            resposta_final = "Ótimo! Se precisar de mais alguma coisa, é só perguntar. 😊"
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            session["mostrar_3_arquivos"] = False
            session["resultados_sem"] = []
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=[],
                                   tempo=tempo_resposta,
                                   exibir_botoes=False,
                                   mostrar_3_arquivos=False)

        if pergunta.lower() == "👍 sim, me envie os outros três resultados mais próximos":
            resposta_final = "Aqui estão os outros três documentos relevantes que encontrei:"
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            session["mostrar_3_arquivos"] = True
            exibir_botoes = True
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=resultados_sem,
                                   tempo=tempo_resposta,
                                   exibir_botoes=True,
                                   mostrar_3_arquivos=True)

        if pergunta.lower() == "👍 sim, atenderam.":
            resposta_final = "Que bom que os documentos atenderam sua necessidade! 😊 Se precisar de mais alguma coisa, estou à disposição."
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            session["mostrar_3_arquivos"] = False
            session["resultados_sem"] = []
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=[],
                                   tempo=tempo_resposta,
                                   exibir_botoes=False,
                                   mostrar_3_arquivos=False)

        if pergunta.lower() == "🔄 não, quero refazer a busca":
            resposta_final = "Certo, por favor, digite sua nova pergunta."
            chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final
            })
            session["chat_history"] = chat_history
            session["mostrar_3_arquivos"] = False
            session["resultados_sem"] = []
            tempo_resposta = 0.01
            return render_template("index.html",
                                   chat_history=chat_history,
                                   pergunta_atual=pergunta,
                                   resposta_atual=resposta_final,
                                   resultados_sem=[],
                                   tempo=tempo_resposta,
                                   exibir_botoes=False,
                                   mostrar_3_arquivos=False)

        if pergunta:
            inicio = time.time()
            try:
                embedding_pergunta = modelo.encode(pergunta, convert_to_numpy=True)
                k = 5
                D, I = faiss_index.search(np.array([embedding_pergunta]), k)

                documentos_relevantes = []

                threshold = 2.0

                for distancia, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    if distancia > threshold:
                        continue

                    doc_meta = metadados[idx]
                    arquivo_pdf = doc_meta.get("arquivo", "")
                    arquivo_pdf_normalizado = normalizar_nome(arquivo_pdf)
                    info_json = mapa_arquivos.get(arquivo_pdf_normalizado)

                    if info_json:
                        titulo = info_json["titulo"]
                        link = info_json["link"]
                    else:
                        titulo = arquivo_pdf
                        link = "#"

                    documentos_relevantes.append({
                        "nome": titulo,
                        "link": link
                    })

                session["resultados_sem"] = documentos_relevantes

                if not documentos_relevantes:
                    prompt = f"""
Você é um assistente especializado em legislação do PPGCAP.

A pergunta do usuário foi: "{pergunta}"

Nenhum documento relevante foi encontrado para essa pergunta.

IMPORTANTE:
- Responda com a seguinte frase: "Não encontrei nenhum documento sobre esse assunto. Você poderia fornecer mais detalhes sobre o que procura?"
- NÃO tente inventar documentos ou informações.

Sua resposta:
"""
                    exibir_botoes = False
                    session["mostrar_3_arquivos"] = False

                else:
                    doc_principal = documentos_relevantes[0]
                    conteudo_docs_texto = f"Título: {doc_principal['nome']}\nLink: {doc_principal['link']}\n"

                    prompt = f"""
Você é um assistente especializado em legislação do PPGCAP.

Seu papel é ajudar o usuário a encontrar informações nas resoluções oficiais do programa.

IMPORTANTE:
- Responda apenas com base no documento principal listado abaixo.
- NÃO cite documentos que não estejam listados.
- NÃO invente títulos ou links.
- Se a pergunta do usuário não estiver relacionada às legislações do PPGCAP, diga: "Não posso falar nada sobre esse assunto."

Aqui está a pergunta do usuário: "{pergunta}"

Documento principal relevante encontrado:

{conteudo_docs_texto}

Além disso, encontrei outros três arquivos que podem ser relevantes por conter o trecho "{pergunta}".

Finalize sua resposta com:
"Deseja que eu lhe envie?"

Não envie os 3 arquivos ainda. Apenas aguarde a resposta do usuário.
"""
                    exibir_botoes = True
                    session["mostrar_3_arquivos"] = False

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
                           resultados_sem=session["resultados_sem"],
                           tempo=tempo_resposta,
                           exibir_botoes=exibir_botoes,
                           mostrar_3_arquivos=session["mostrar_3_arquivos"])

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    session.pop("resultados_sem", None)
    session.pop("mostrar_3_arquivos", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)