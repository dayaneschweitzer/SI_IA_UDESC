import os
import pickle
import json
import re
import unicodedata
import numpy as np
from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.api_core.client_options import ClientOptions
from google.ai.generativelanguage_v1beta.types import Content, Part, EmbedContentRequest, TaskType

api_key = "AIzaSyDl1NxUz3X2893pXIMnzZFH41XXYVw6kSU"
client = GenerativeServiceClient(client_options=ClientOptions(api_key=api_key))

def gerar_embedding(texto):
    req = EmbedContentRequest(
        model="models/embedding-001",
        content=Content(parts=[Part(text=texto)]),
        task_type=TaskType.RETRIEVAL_DOCUMENT
    )
    resp = client.embed_content(req)
    return resp.embedding.values

def normalizar_nome(nome):
    n = unicodedata.normalize('NFD', nome.lower()).encode('ascii', 'ignore').decode('utf-8')
    return n.replace('ç','c').replace('_','').replace('-','').replace(' ','')
def extrair_numero_ano(nome):
    m = re.search(r'resolucao(\d{3})(\d{4})', nome.lower())
    return f"resolucao{m.group(1)}{m.group(2)}" if m else None

os.makedirs("vetores", exist_ok=True)
with open("PDFs_Udesc/resolucoes_ppgcap.json",'r',encoding='utf-8') as f:
    info = json.load(f)
prefixos = {}
for i in info:
    p = extrair_numero_ano(normalizar_nome(i["arquivo"]))
    if not p and "titulo" in i:
        p = extrair_numero_ano(normalizar_nome(i["titulo"]))
    if p:
        prefixos[p] = i["arquivo"]

embeddings, metadados, full_textos = [], [], {}
for fn in sorted(os.listdir("textos_extraidos")):
    if not fn.endswith(".txt"): continue
    with open(os.path.join("textos_extraidos",fn),'r',encoding='utf-8') as f:
        texto = f.read().strip()
    pfn = extrair_numero_ano(normalizar_nome(re.sub(r'_chunk_\d+\.txt$','',fn)))
    if not pfn or pfn not in prefixos: continue
    pdf = prefixos[pfn]
    d = next((x for x in info if x["arquivo"]==pdf),{})
    cid = int(re.search(r'_chunk_(\d+)\.txt$',fn).group(1)) if "_chunk_" in fn else 0

    emb = gerar_embedding(texto)
    embeddings.append(emb)
    metadados.append({"arquivo":pdf,"chunk_id":cid,"titulo":d.get("titulo",pdf),"link":d.get("link"),"texto":texto})
    full_textos[pdf] = full_textos.get(pdf,"")+("\n\n" if full_textos.get(pdf) else "")+texto

emb_np = np.array(embeddings)
np.save("vetores/embeddings.npy",emb_np)
with open("vetores/metadados.pkl","wb") as f: pickle.dump(metadados,f)
with open("vetores/full_textos.pkl","wb") as f: pickle.dump(full_textos,f)

print(f"Gerados {len(embeddings)} embeddings, dimensão {emb_np.shape[1]}")
