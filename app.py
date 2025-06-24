import os
import re
import pickle
import numpy as np
import shelve
from hashlib import sha256
from flask import Flask, request, render_template, session
from sentence_transformers import SentenceTransformer
import requests

app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta"

with open("vetores/metadados.pkl", "rb") as f:
    metadados = pickle.load(f)
embeddings = np.load("vetores/norm_embeddings.npy")
modelo = SentenceTransformer("intfloat/e5-base-v2")

def gerar_embedding(texto):
    entrada = f"query: {texto}"
    try:
        with shelve.open("vetores/pergunta_cache", flag='c') as cache:
            if entrada in cache:
                return np.array(cache[entrada])
            emb = modelo.encode(entrada, convert_to_numpy=True)
            cache[entrada] = emb.tolist()
            return emb
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return np.zeros(embeddings.shape[1])

def buscar_resolucao_por_numero(pergunta):
    match = re.search(r"\b(\d{3})/(\d{4})\b", pergunta)
    if not match:
        return None
    numero = f"{match.group(1)}/{match.group(2)}"
    for doc in metadados:
        if numero in doc["titulo"] or numero in doc["texto"]:
            return numero, doc
    return None

def selecionar_documentos(pergunta):
    qe = gerar_embedding(pergunta)
    scores = np.dot(embeddings, qe) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qe) + 1e-10)
    idxs = np.argsort(scores)[-5:][::-1]
    docs = [metadados[i] for i in idxs]
    return docs

def gerar_resposta_mistral(pergunta, docs):
    contexto = "\n".join(f"- {d['titulo']}" for d in docs)
    prompt = (
        "Você é um assistente especializado nas resoluções administrativas do PPGCAP da UDESC.\n"
        "Com base nessa lista de títulos, diga apenas o título mais relevante que responde à pergunta.\n"
        "Se o usuário mencionar o número de resolução (ex: '047/2025'), devolva esse título se existir.\n"
        "Se não houver correspondência, diga 'nenhum documento claro encontrado'.\n\n"
        f"Títulos:\n{contexto}\n\n"
        f"Pergunta: {pergunta}\n"
        "Resposta:"
    )
    h = sha256(prompt.encode()).hexdigest()
    try:
        with shelve.open("vetores/resposta_cache", flag='c') as cache:
            if h in cache:
                return cache[h]
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            titulo = resp.json().get("response", "").strip() if resp.status_code == 200 else ""
            cache[h] = titulo
            return titulo
    except Exception:
        return ""

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = [{
            "pergunta": "",
            "resposta": "Olá! Sou um assistente especializado nas resoluções do PPGCAP. Como posso te ajudar?"
        }]

    pergunta = request.form.get("pergunta", "").strip() if request.method == "POST" else ""
    resposta = ""

    if pergunta:
        pl = pergunta.lower()

        if any(p in pl for p in ["olá", "oi", "bom dia", "boa tarde", "boa noite"]):
            resposta = "Olá, espero que você esteja bem! :) Como posso ajudar você hoje?"

        elif any(p in pl for p in [
            "quero morrer", "me matar", "tirar minha vida", "acabar com tudo", "não aguento mais",
            "suic", "morrer", "não quero viver", "sumir", "desistir da vida"
        ]):
            resposta = (
                'Encontre ajuda com o CVV: <a href="https://cvv.org.br/" target="_blank">cvv.org.br</a> ou ligue 188.<br>'
                'Ajudo com algo mais?'
            )

        elif any(p in pl for p in [
            "mulher", "gay", "preto", "piada", "presidente", "data de hoje", "branco", "negro", "trans",
            "quem é", "quantos anos", "sabe programar", "idiota", "burro", "ladrão", "lgbt", "religião"
        ]):
            resposta = (
                "Desculpe, não posso falar nada sobre isso. "
                "Posso ajudar com resoluções e portarias. Precisa?"
            )

        else:
            num_info = buscar_resolucao_por_numero(pergunta)
            if num_info:
                numero, doc = num_info
                resposta = (
                    f"As informações sobre <strong>{pergunta}</strong> podem ser encontradas no seguinte documento:<br>"
                    f'<a href="{doc["link"]}" target="_blank">{doc["titulo"]}</a><br><br>'
                    "Ajudo com algo mais?"
                )
            else:
                docs = selecionar_documentos(pergunta)
                titulo_sel = gerar_resposta_mistral(pergunta, docs)
                principal = next((d for d in docs if titulo_sel.lower() in d["titulo"].lower()), docs[0])
                resposta = (
                    f"As informações sobre <strong>{pergunta}</strong> podem ser encontradas no seguinte documento:<br>"
                    f'<a href="{principal["link"]}" target="_blank">{principal["titulo"]}</a><br><br>'
                )
                outros = [d for d in docs if d != principal]
                if outros:
                    resposta += "Encontrei outros documentos que podem ser relevantes sobre esse tema:<br>"
                    for d in outros:
                        resposta += f'<a href="{d["link"]}" target="_blank">{d["titulo"]}</a><br>'
                    resposta += "Ajudo com algo mais?"
                else:
                    resposta += (
                        "Esse documento lhe ajuda ou prefere que eu refaça a busca? :)"
                    )

        session["chat_history"].append({"pergunta": pergunta, "resposta": resposta})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)
