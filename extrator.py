import fitz
import os

def extrair_texto_pdf(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    texto_total = ""
    for pagina in doc:
        texto_total += pagina.get_text()
    doc.close()
    return texto_total

def dividir_em_chunks(texto, tamanho_max=500):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), tamanho_max):
        chunk = " ".join(palavras[i:i + tamanho_max])
        chunks.append(chunk)
    return chunks

def processar_pdfs(pasta_pdfs="portarias", ignorar=[]):
    os.makedirs("textos_extraidos", exist_ok=True)
    erros = []

    for arquivo in os.listdir(pasta_pdfs):
        if arquivo.endswith(".pdf") and arquivo not in ignorar:
            caminho = os.path.join(pasta_pdfs, arquivo)
            try:
                texto = extrair_texto_pdf(caminho)
                chunks = dividir_em_chunks(texto, tamanho_max=500)
                for i, chunk in enumerate(chunks):
                    nome_txt = arquivo.replace(".pdf", f"_chunk{i}.txt")
                    with open(os.path.join("textos_extraidos", nome_txt), "w", encoding="utf-8") as f:
                        f.write(chunk)
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")
                erros.append(arquivo)
    return erros
