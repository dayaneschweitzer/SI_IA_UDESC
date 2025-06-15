import os
import pickle
import numpy as np
import json
import re
import unicodedata
from sentence_transformers import SentenceTransformer

def normalizar_nome(nome):
    nome = nome.lower()
    nome = unicodedata.normalize('NFD', nome).encode('ascii', 'ignore').decode('utf-8')
    nome = nome.replace('ç', 'c')
    nome = nome.replace('_', '').replace('-', '').replace(' ', '')
    return nome

def extrair_numero_ano(nome):
    nome = nome.lower()
    match = re.search(r'resolucao(\d{3})(\d{4})', nome)
    if match:
        numero = match.group(1)
        ano = match.group(2)
        return f'resolucao{numero}{ano}'
    return None

def legal_chunk_text(text, max_chunk_size=1000):
    patterns = [
        "dispõe sobre", "regulamenta", "estabelece", "altera a resolução",
        "revoga", "aprova", "institui", "cria", "fixa", "define", "retifica"
    ]

    lines = text.split("\n")
    header_lines = []
    body_lines = []
    found_header_marker = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(p in stripped.lower() for p in patterns) and not found_header_marker:
            header_lines.append(stripped)
            found_header_marker = True
        elif not found_header_marker or len(header_lines) < 3:
            header_lines.append(stripped)
        else:
            body_lines.append(stripped)

    if not body_lines:
        body_lines = lines[len(header_lines):] if len(lines) > len(header_lines) else []

    header_text = "\n".join(header_lines)
    body_text = "\n".join(body_lines)

    paragraphs = re.split(r'\n\s*\n', body_text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    if not chunks:
        fallback_chunk = header_text + "\n\n" + body_text
        chunks = [fallback_chunk.strip()]

    return header_text, chunks

textos_folder = "textos_extraidos"
vetores_folder = "vetores"
pdf_base_path = "PDFs_Udesc"

os.makedirs(vetores_folder, exist_ok=True)

with open(os.path.join(pdf_base_path, "resolucoes_ppgcap.json"), 'r', encoding='utf-8') as f:
    resolucoes_info = json.load(f)

prefixo_para_arquivo_real = {}
for item in resolucoes_info:
    arquivo_json = item["arquivo"]
    prefixo = normalizar_nome(arquivo_json)
    prefixo = extrair_numero_ano(prefixo)
    if not prefixo and "titulo" in item:
        titulo = normalizar_nome(item["titulo"])
        prefixo = extrair_numero_ano(titulo)
        if prefixo:
            print(f"[INFO] Usando fallback pelo título '{item['titulo']}' → mapeado como '{prefixo}'.")
    if prefixo:
        prefixo_para_arquivo_real[prefixo] = arquivo_json
    else:
        print(f"[WARNING] Não consegui extrair numero/ano de '{arquivo_json}' nem do título — pode não bater.")

modelo_nome = "all-mpnet-base-v2"
modelo = SentenceTransformer(modelo_nome)

embeddings = []
metadados = []
full_textos = {}

for filename in os.listdir(textos_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(textos_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()

        match_chunk_manual = re.search(r'_chunk_(\d+)\.txt$', filename)
        is_manual_chunk = match_chunk_manual is not None

        prefixo_filename = re.sub(r'_chunk_\d+\.txt$', '', filename)
        prefixo_filename = normalizar_nome(prefixo_filename)
        prefixo_filename = extrair_numero_ano(prefixo_filename)
        if not prefixo_filename:
            print(f"[WARNING] Não consegui extrair numero/ano de '{filename}' — este chunk será ignorado.")
            continue

        arquivo_real_pdf = prefixo_para_arquivo_real.get(prefixo_filename, None)

        if not arquivo_real_pdf:
            print(f"[WARNING] Não achei correspondência para '{prefixo_filename}' — este chunk será ignorado.")
            continue

        info_doc = next((item for item in resolucoes_info if item["arquivo"] == arquivo_real_pdf), {})
        titulo = info_doc.get("titulo", arquivo_real_pdf)
        link = info_doc.get("link", None)

        if is_manual_chunk:
            chunk_id = int(match_chunk_manual.group(1))
            chunk = texto.strip()
            embedding = modelo.encode(chunk, convert_to_numpy=True)
            embeddings.append(embedding)
            metadados.append({
                "arquivo": arquivo_real_pdf,
                "chunk_id": chunk_id,
                "titulo": titulo,
                "link": link,
                "texto": chunk
            })
            if arquivo_real_pdf not in full_textos:
                full_textos[arquivo_real_pdf] = chunk
            else:
                full_textos[arquivo_real_pdf] += "\n\n" + chunk

        else:
            header_text, chunks = legal_chunk_text(texto, max_chunk_size=1000)
            full_textos[arquivo_real_pdf] = header_text + "\n\n" + "\n".join(chunks)
            for i, chunk in enumerate(chunks):
                embedding = modelo.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
                metadados.append({
                    "arquivo": arquivo_real_pdf,
                    "chunk_id": i,
                    "titulo": titulo,
                    "link": link,
                    "texto": chunk
                })

embeddings = np.array(embeddings)
if embeddings.shape[0] == 0:
    print("[ERROR] Nenhum embedding foi gerado. Verifique os textos ou a função de chunking.")
    exit(1)

dim = embeddings.shape[1]

with open(os.path.join(vetores_folder, "metadados.pkl"), 'wb') as f:
    pickle.dump(metadados, f)

with open(os.path.join(vetores_folder, "full_textos.pkl"), 'wb') as f:
    pickle.dump(full_textos, f)

print(f"\n[DEBUG] Sanity check")
print(f"[DEBUG] Total embeddings: {len(embeddings)}")
print(f"[DEBUG] Total metadados  : {len(metadados)}")

print(f"\n Gerados {len(embeddings)} embeddings e índice FAISS salvo.")
print(f"Arquivos salvos: faiss_index.index, metadados.pkl, full_textos.pkl")
