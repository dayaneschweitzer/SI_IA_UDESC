import os
import pickle
import numpy as np
import re
from flask import Flask, request, render_template, session
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

load_dotenv()
api_key = "AIzaSyDl1NxUz3X2893pXIMnzZFH41XXYVw6kSU"
model = GenerativeModel(model_name="models/gemini-1.5-pro")

app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta"

with open("vetores/metadados.pkl", "rb") as f:
    metadados = pickle.load(f)
embeddings = np.load("vetores/embeddings.npy")

def gerar_embedding(texto):
    from google.ai.generativelanguage_v1beta import EmbedContentRequest, Content, Part
    from google.ai.generativelanguage_v1beta.services.generative_service import GenerativeServiceClient
    from google.api_core.client_options import ClientOptions
    from google.ai.generativelanguage_v1beta.types import TaskType

    client = GenerativeServiceClient(client_options=ClientOptions(api_key=api_key))
    req = EmbedContentRequest(
        model="models/embedding-001",
        content=Content(parts=[Part(text=texto)]),
        task_type=TaskType.RETRIEVAL_QUERY
    )
    resp = client.embed_content(req)
    return np.array(resp.embedding.values)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = [{"pergunta": "", "resposta": "Olá! Sou um assistente especializado nas resoluções do PPGCAP. Como posso te ajudar?"}]

    pergunta = request.form.get("pergunta", "").strip() if request.method == "POST" else ""
    resposta_final = ""

    saudacoes = re.compile(r"\b(oi|olá|opa|bom dia|boa tarde|boa noite)\b", re.IGNORECASE)
    ofensivo = re.compile(r"\b(viado|idiota|burro|lixo|besta|palavr[aã]o\d*|preto|negro|mulher.*não sabe|gay|lésbica|trans|marginal)\b", re.IGNORECASE)
    suicidio = re.compile(r"\b(suic[ií]dio|me matar|tirar minha vida|quero morrer)\b", re.IGNORECASE)
    irrelevante = re.compile(r"\b(presidente|piada|que dia|data de hoje|qual seu nome|quem é você|filme|notícia|clima|tempo)\b", re.IGNORECASE)

    if pergunta:
        if saudacoes.search(pergunta):
            resposta_final = "Olá! Espero que você esteja bem! :)\nComo posso ajudar você hoje?"
        elif ofensivo.search(pergunta) or irrelevante.search(pergunta):
            resposta_final = "Desculpe, não posso falar nada a respeito disso. Eu só posso ajudar com a busca por resoluções e portarias do PPGCAP. Precisa de ajuda nesse sentido?"
        elif suicidio.search(pergunta):
            resposta_final = (
                "Encontre ajuda falando com um dos voluntários do Centro de Valorização da Vida - CVV "
                "👉 <a href=\"https://cvv.org.br/\" target=\"_blank\">cvv.org.br</a><br>"
                "Ou ligue para o número 188 (Gratuitamente)<br>"
                "Ajudo com algo a mais?"
            )
        else:
            query_embedding = gerar_embedding(pergunta)
            scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10)
            top_k_indices = np.argsort(scores)[-5:][::-1]
            principais = top_k_indices[0]
            relevantes = top_k_indices[1:]

            doc_principal = metadados[principais]
            resposta_final = f"As informações sobre {pergunta} podem ser encontradas no seguinte documento:<br>"
            resposta_final += f"<a href=\"{doc_principal['link']}\" target=\"_blank\">{doc_principal['titulo']}</a><br><br>"
            resposta_final += f"Encontrei outros documentos que podem ser relevantes sobre esse tema:<br>"

            for idx in relevantes:
                doc = metadados[idx]
                resposta_final += f"<a href=\"{doc['link']}\" target=\"_blank\">{doc['titulo']}</a><br>"

            resposta_final += "Ajudo com algo a mais?"

        session["chat_history"].append({
            "pergunta": pergunta,
            "resposta": resposta_final
        })
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)
