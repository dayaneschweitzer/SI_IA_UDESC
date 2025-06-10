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

    chat_history = session["chat_history"]
    pergunta = ""
    resposta_final = ""
    tempo_resposta = 0
    resultados_sem = []
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
                                   exibir_botoes=False)

        if pergunta.lower() == "sim, atenderam.":
            resposta_final = "Ótimo! Se precisar de mais alguma coisa, é só perguntar. 😊"
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
                                   exibir_botoes=False)

        if pergunta.lower() == "não, quero refazer a busca":
            resposta_final = "Certo, por favor, digite sua nova pergunta."
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
                                   exibir_botoes=False)

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

                    trecho_resumido = doc_meta["texto"][:300].replace("\n", " ").strip() + "..."

                    documentos_relevantes.append({
                        "nome": titulo,
                        "link": link,
                        "trecho": trecho_resumido
                    })

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
                else:
                    conteudo_docs_texto = ""
                    for i, doc in enumerate(documentos_relevantes):
                        conteudo_docs_texto += f"Documento {i+1}:\n"
                        conteudo_docs_texto += f"Título: {doc['nome']}\n"
                        conteudo_docs_texto += f"Link: {doc['link']}\n"
                        conteudo_docs_texto += f"Trecho: {doc['trecho']}\n\n"

                    prompt = f"""
Você é um assistente especializado em legislação do PPGCAP.

Seu papel é ajudar o usuário a encontrar informações nas resoluções oficiais do programa.

IMPORTANTE:
- Responda apenas com base nos documentos listados abaixo.
- NÃO cite documentos que não estejam listados.
- NÃO invente títulos ou links.
- Se a pergunta do usuário não estiver relacionada às legislações do PPGCAP, diga: "Não posso falar nada sobre esse assunto."

Aqui está a pergunta do usuário: "{pergunta}"

Documentos relevantes encontrados:

{conteudo_docs_texto}

Formato de resposta que você deve seguir:

Aqui está um documento relevante com sua busca:

Título: [Título do documento mais relevante]
Link: [Link correspondente]

Além disso, encontrei outros três arquivos que podem ser relevantes por conter o trecho "{pergunta}":

[Listar os três documentos com Título + Link + pequeno trecho]

Finalizar com a pergunta:
Esses documentos lhe atendem?
"""
                    resultados_sem = documentos_relevantes
                    exibir_botoes = True

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
                           tempo=tempo_resposta,
                           exibir_botoes=exibir_botoes)

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)
