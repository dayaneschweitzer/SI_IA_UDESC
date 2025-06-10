import os
import pickle
import faiss
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
    nome = nome.replace('aprovadaresolucao', 'resolucao')
    nome = nome.replace('minutaresolucao', 'resolucao')
    match = re.search(r'resolucao[_\-]?(\d{3})[_\-]?(\d{4})', nome)
    if match:
        numero = match.group(1)
        ano = match.group(2)
        return f'resolucao{numero}{ano}'
    match = re.search(r'resolucao[_\-]?(\d{3})', nome)
    if match:
        numero = match.group(1)
        return f'resolucao{numero}'
    return None

def chunk_text(text, max_chunk_size=1000):
    paragraphs = re.split(r'\n\s*\n', text)
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
    return chunks

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

for filename in os.listdir(textos_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(textos_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            texto = f.read()
            chunks = chunk_text(texto, max_chunk_size=1000)
            prefixo_filename = re.sub(r'_chunk_\d+\.txt$', '', filename)
            prefixo_filename = normalizar_nome(prefixo_filename)
            prefixo_filename = extrair_numero_ano(prefixo_filename)
            if not prefixo_filename:
                print(f"[WARNING] Não consegui extrair numero/ano de '{filename}' — este chunk será ignorado.")
                continue
            arquivo_real_pdf = prefixo_para_arquivo_real.get(prefixo_filename, None)
            if not arquivo_real_pdf and prefixo_filename.startswith('resolucao') and len(prefixo_filename) == len('resolucaoXXXYYYY'):
                prefixo_fallback = prefixo_filename[:len('resolucaoXXX')]
                arquivo_real_pdf = prefixo_para_arquivo_real.get(prefixo_fallback, None)
                if arquivo_real_pdf:
                    print(f"[INFO] Usando fallback para '{prefixo_fallback}' → mapeado com '{arquivo_real_pdf}'.")
            if not arquivo_real_pdf:
                print(f"[WARNING] Não achei correspondência para '{prefixo_filename}' — este chunk será ignorado.")
                continue
            info_doc = next((item for item in resolucoes_info if item["arquivo"] == arquivo_real_pdf), {})
            titulo = info_doc.get("titulo", arquivo_real_pdf)
            link = info_doc.get("link", None)
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
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, os.path.join(vetores_folder, "faiss_index.index"))

with open(os.path.join(vetores_folder, "metadados.pkl"), 'wb') as f:
    pickle.dump(metadados, f)

print(f"\n Gerados {len(embeddings)} chunks e índice FAISS salvo.")
print(f" metadados.pkl com 'arquivo' igual ao JSON.")
