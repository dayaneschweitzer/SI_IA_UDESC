import os
import pickle
import numpy as np
import shelve
from hashlib import sha256
from flask import Flask, request, render_template, session
from sentence_transformers import SentenceTransformer
import requests

# Inicializar Flask
app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta"

# Carregar arquivos
with open("vetores/metadados.pkl", "rb") as f:
    metadados = pickle.load(f)
embeddings = np.load("vetores/norm_embeddings.npy")

# Carregar modelo de embeddings
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
        return np.zeros(768)

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

def gerar_resposta_mistral(pergunta, documentos):
    contexto = ""
    for doc in documentos:
        contexto += f"{doc['titulo']}:\n{doc['texto']}\n\n"

    prompt = (
        "Você é um assistente especializado nas resoluções administrativas do PPGCAP da UDESC. "
        "Com base nos documentos abaixo, responda com precisão e clareza à pergunta fornecida. "
        "Se a resposta não estiver claramente presente, diga que não encontrou essa informação nos documentos.\n\n"
        f"Documentos:\n{contexto}\n"
        f"Pergunta: {pergunta}\n"
        "Resposta:"
    )

    prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()

    try:
        with shelve.open("vetores/resposta_cache", flag='c') as cache:
            if prompt_hash in cache:
                return cache[prompt_hash]

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                resposta = response.json()["response"].strip()
                cache[prompt_hash] = resposta
            else:
                resposta = f"Erro {response.status_code} ao consultar o modelo Mistral via Ollama."
    except Exception as e:
        resposta = f"Erro ao conectar com Ollama: {e}"

    return resposta

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
            documentos_contexto = [principal] + outros
            resposta_gerada = gerar_resposta_mistral(pergunta, documentos_contexto)

            resposta = f"{resposta_gerada}<br><br>"
            resposta += f'Documento principal: <a href="{principal["link"]}" target="_blank">{principal["titulo"]}</a><br>'
            if outros:
                resposta += "Outros documentos relevantes:<br>"
                for doc in outros:
                    resposta += f'<a href="{doc["link"]}" target="_blank">{doc["titulo"]}</a><br>'

        session["chat_history"].append({"pergunta": pergunta, "resposta": resposta})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)