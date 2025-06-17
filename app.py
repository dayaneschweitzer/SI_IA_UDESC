import os
import pickle
import numpy as np
from flask import Flask, request, render_template, session
from google.ai.generativelanguage_v1beta import GenerativeServiceClient, Content, Part, EmbedContentRequest, GenerateContentRequest, TaskType
from google.api_core.client_options import ClientOptions

api_key = "AIzaSyDl1NxUz3X2893pXIMnzZFH41XXYVw6kSU"
client = GenerativeServiceClient(client_options=ClientOptions(api_key=api_key))

app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta"

with open("vetores/metadados.pkl", "rb") as f:
    metadados = pickle.load(f)
embeddings = np.load("vetores/embeddings.npy")

def gerar_embedding(texto):
    req = EmbedContentRequest(
        model="models/embedding-001",
        content=Content(parts=[Part(text=texto)]),
        task_type=TaskType.RETRIEVAL_QUERY
    )
    resp = client.embed_content(req)
    return np.array(resp.embedding.values)

def selecionar_documentos(pergunta):
    qe = gerar_embedding(pergunta)
    scores = np.dot(embeddings, qe) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qe) + 1e-10)
    top_idxs = np.argsort(scores)[-10:][::-1]

    idx0 = top_idxs[0]
    doc0 = metadados[idx0]
    usados = {doc0["link"]}

    outros = []
    for idx in top_idxs[1:]:
        link = metadados[idx]["link"]
        if link and link not in usados:
            outros.append(metadados[idx])
            usados.add(link)
        if len(outros) >= 4:
            break

    return doc0, outros

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = [{"pergunta": "", "resposta": "Olá! Sou um assistente especializado nas resoluções do PPGCAP. Como posso te ajudar?"}]

    pergunta = request.form.get("pergunta", "").strip() if request.method == "POST" else ""
    resposta = ""

    if pergunta:
        pergunta_lower = pergunta.lower()

        if any(p in pergunta_lower for p in ["olá", "oi", "bom dia", "boa tarde", "boa noite"]):
            resposta = "Olá, espero que você esteja bem! :) Como posso ajudar você hoje?"

        elif any(p in pergunta_lower for p in [
            "quero morrer", "me matar", "tirar minha vida", "acabar com tudo", "não aguento mais",
            "suic", "morrer", "não quero viver", "sumir", "desistir da vida"
        ]):
            resposta = (
                'Encontre ajuda com um dos voluntários do Centro de Valorização da Vida - '
                '<a href="https://cvv.org.br/" target="_blank">cvv.org.br</a><br>'
                'Ou ligue 188 (Gratuito)<br>Ajudo com algo a mais?'
            )

        elif any(p in pergunta_lower for p in [
            "mulher", "gay", "preto", "piada", "presidente", "data de hoje", "branco", "negro", "trans",
            "quem é", "quantos anos", "sabe programar", "idiota", "burro", "ladrão", "lgbt", "religião"
        ]):
            resposta = (
                "Desculpe, não posso falar nada a respeito disso. "
                "Só posso ajudar com a busca por resoluções e portarias do PPGCAP. "
                "Precisa de ajuda nesse sentido?"
            )

        else:
            principal, outros = selecionar_documentos(pergunta)
            tp, lp = principal["titulo"], principal["link"]

            resposta = (f"As informações sobre <strong>{pergunta}</strong> podem ser encontradas no seguinte documento:<br>"
                        f"<a href=\"{lp}\" target=\"_blank\">{tp}</a><br><br>")

            if outros:
                resposta += "Encontrei outros documentos que podem ser relevantes sobre esse tema:<br>"
                for doc in outros:
                    resposta += f'<a href="{doc["link"]}" target="_blank">{doc["titulo"]}</a><br>'
                resposta += "Ajudo com algo mais?"
            else:
                resposta += ("Esse documento lhe ajuda ou prefere que eu refaça a busca? "
                              "É só você me dar mais informações do que procura que eu posso lhe ajudar! :)")

        session["chat_history"].append({"pergunta": pergunta, "resposta": resposta})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)