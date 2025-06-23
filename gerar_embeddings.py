import os
import pickle
import json
import re
import unicodedata
import numpy as np
import requests
import shelve

def gerar_embedding(texto):
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": texto}
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Erro {response.status_code} ao gerar embedding.")
            return np.zeros(768)
    except Exception as e:
        print(f"Erro ao conectar com Ollama: {e}")
        return np.zeros(768)

def normalizar_nome(nome):
    n = unicodedata.normalize('NFD', nome.lower()).encode('ascii', 'ignore').decode('utf-8')
    return n.replace('ç','c').replace('_','').replace('-','').replace(' ','')

def extrair_numero_ano(nome):
    m = re.search(r'resolucao(\d{3})(\d{4})', nome.lower())
    return f"resolucao{m.group(1)}{m.group(2)}" if m else None

# Criar diretório de saída
os.makedirs("vetores", exist_ok=True)

# Carregar metadados dos PDFs
with open("PDFs_Udesc/resolucoes_ppgcap.json", 'r', encoding='utf-8') as f:
    info = json.load(f)

# Mapear prefixos
prefixos = {}
for i in info:
    p = extrair_numero_ano(normalizar_nome(i["arquivo"]))
    if not p and "titulo" in i:
        p = extrair_numero_ano(normalizar_nome(i["titulo"]))
    if p:
        prefixos[p] = i["arquivo"]

# Gerar embeddings
embeddings, metadados, full_textos = [], [], {}
for fn in sorted(os.listdir("textos_extraidos")):
    if not fn.endswith(".txt"):
        continue
    with open(os.path.join("textos_extraidos", fn), 'r', encoding='utf-8') as f:
        texto = f.read().strip()

    pfn = extrair_numero_ano(normalizar_nome(re.sub(r'_chunk_\d+\.txt$', '', fn)))
    if not pfn or pfn not in prefixos:
        continue

    pdf = prefixos[pfn]
    d = next((x for x in info if x["arquivo"] == pdf), {})
    cid = int(re.search(r'_chunk_(\d+)\.txt$', fn).group(1)) if "_chunk_" in fn else 0

    emb = gerar_embedding(texto)
    embeddings.append(emb)
    metadados.append({
        "arquivo": pdf,
        "chunk_id": cid,
        "titulo": d.get("titulo", pdf),
        "link": d.get("link"),
        "texto": texto
    })
    full_textos[pdf] = full_textos.get(pdf, "") + ("\n\n" if full_textos.get(pdf) else "") + texto

# Normalizar e salvar embeddings
emb_np = np.array(embeddings)
norm_embeddings = emb_np / np.linalg.norm(emb_np, axis=1, keepdims=True)
np.save("vetores/norm_embeddings.npy", norm_embeddings)

# Salvar metadados e textos
with open("vetores/metadados.pkl", "wb") as f:
    pickle.dump(metadados, f)
with open("vetores/full_textos.pkl", "wb") as f:
    pickle.dump(full_textos, f)

# Criar cache de embeddings de perguntas
with shelve.open("vetores/pergunta_cache") as cache:
    cache.clear()

print(f"Gerados {len(embeddings)} embeddings, dimensão {emb_np.shape[1]}")
print("Embeddings normalizados salvos em 'vetores/norm_embeddings.npy'")
print("Cache de perguntas inicializado em 'vetores/pergunta_cache'")
