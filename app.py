import os
import pickle
import json
import unicodedata
from flask import Flask, request, render_template, session
from langchain_ollama import OllamaLLM

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = texto.replace('ç', 'c')
    texto = texto.replace('-', ' ')
    texto = texto.replace('_', ' ')
    texto = ' '.join(texto.split()) 
    return texto

def detectou_conteudo_inadequado(pergunta):
    texto = normalizar_texto(pergunta)
    palavroes = ["droga", "puta", "são ruim", "são ruins", "mulheres", "não sabem", "gays", "negros", "pretos", "sexo", "sexualidade", "lésbica"]
    aleatorias = ["rico", "hackear", "loteria", "uma piada", "uma historia"]
    pejorativas = ["é feio", "matar", "morte aos", "odeio", "bichas", "bixas", "tiro", "tiros", "lgbt"]
    saudacoes = ["oi", "ola", "bom dia", "boa tarde", "boa noite"]

    for termo in saudacoes:
        if termo in texto:
            return "saudacao"
    for termo in palavroes + aleatorias + pejorativas:
        if termo in texto:
            return "inadequado"
    return "normal"

pdf_base_path = os.path.abspath("PDFs_Udesc")
vetores_folder = os.path.abspath("vetores")

app = Flask(__name__, template_folder='.')
app.secret_key = "chave_super_secreta_para_sessao"

with open(os.path.join(pdf_base_path, "resolucoes_ppgcap.json"), 'r', encoding='utf-8') as f:
    resolucoes_info = json.load(f)

with open(os.path.join(vetores_folder, "full_textos.pkl"), 'rb') as f:
    full_textos = pickle.load(f)

MODEL_NAME = "mistral"
llm = OllamaLLM(model=MODEL_NAME)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = [{"pergunta": "", "resposta": "Olá! Sou um assistente especializado nas resoluções do PPGCAP. Como posso te ajudar?"}]

    chat_history = session["chat_history"]
    pergunta = ""
    resposta_final = ""
    tempo_resposta = 0

    if request.method == "POST":
        pergunta = request.form.get("pergunta", "").strip()

        if pergunta:
            import time
            inicio = time.time()
            try:
                tipo = detectou_conteudo_inadequado(pergunta)

                if tipo == "saudacao":
                    resposta_final = "Olá! Espero que você esteja bem. Estou aqui para lhe ajudar com as resoluções do PPGCAP. Envie sua pergunta quando quiser."
                elif tipo == "inadequado":
                    resposta_final = "Desculpe, não posso responder a esse tipo de pergunta."
                else:
                    lista_titulos = ""
                    for item in resolucoes_info:
                        titulo = item.get("titulo", "Sem título")
                        link = item.get("link", "#")
                        lista_titulos += f"- {titulo} ({link})\n"

                    prompt_escolha = f"""
Você é um assistente especializado em legislação do PPGCAP.

Abaixo está uma lista de resoluções disponíveis:

{lista_titulos}

Aqui está a pergunta do usuário: "{pergunta}"

Sua tarefa:
- Analise a pergunta e a lista de resoluções.
- Escolha apenas uma resolução da lista que você considera mais relevante para responder à pergunta.
- Copie exatamente o **título como está na lista acima** — não invente, não resuma, não reescreva.
- Se nenhuma for relevante, responda: "Nenhuma resolução é relevante para essa pergunta."
"""
                    resposta_llm_etapa1 = llm.invoke(prompt_escolha)
                    resposta_normalizada = normalizar_texto(resposta_llm_etapa1)

                    documento_escolhido = None
                    for item in resolucoes_info:
                        titulo_normalizado = normalizar_texto(item.get("titulo", ""))
                        if titulo_normalizado in resposta_normalizada or resposta_normalizada in titulo_normalizado:
                            documento_escolhido = item
                            break

                    if not documento_escolhido or "nenhuma resolucao" in resposta_normalizada:
                        resposta_final = "Não consegui identificar qual documento você quer. Por favor, tente novamente com uma pergunta mais específica."
                    else:
                        arquivo_pdf = documento_escolhido.get("arquivo", "")
                        texto_completo = full_textos.get(arquivo_pdf, "")
                        titulo = documento_escolhido.get("titulo", "")
                        link = documento_escolhido.get("link", "#")

                        prompt_resposta = f"""
Você é um assistente especializado em legislação do PPGCAP.

Aqui está a pergunta do usuário: "{pergunta}"

Você deve responder com base no seguinte texto da resolução escolhida:

Título: {titulo}
Link: {link}
Texto completo:

{texto_completo}

IMPORTANTE:
- Responda de forma clara e objetiva.
- Comece com: "As informações sobre [tema da pergunta] pode ser encontrado no seguinte documento:"
- Em seguida, inclua um link HTML no formato: <a href="{link}" target="_blank">{titulo}</a>
- Finalize com: "Ajudo com algo a mais?"
- NÃO repita a pergunta do usuário.
"""

                        resposta_modelo = llm.invoke(prompt_resposta)
                        resposta_final = resposta_modelo

                chat_history.append({
                    "pergunta": pergunta,
                    "resposta": resposta_final
                })
                session["chat_history"] = chat_history

            except Exception as e:
                resposta_final = f"Ocorreu um erro ao processar sua solicitação: {str(e)}"

            tempo_resposta = round(time.time() - inicio, 3)

    return render_template("index.html",
                           chat_history=chat_history,
                           pergunta_atual=pergunta,
                           resposta_atual=resposta_final,
                           tempo=tempo_resposta)

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return "Histórico resetado. <a href='/'>Voltar</a>"

if __name__ == "__main__":
    app.run(debug=True)
