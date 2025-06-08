import json
import os
import fitz 

json_path = "C:/SI_IA_UDESC/PDFs_Udesc/resolucoes_ppgcap.json"
pdf_folder = "C:/SI_IA_UDESC/PDFs_Udesc"
output_folder = "textos_extraidos"

os.makedirs(output_folder, exist_ok=True)

def dividir_em_chunks(texto, tamanho_chunk=300):
    palavras = texto.split()
    chunks = [' '.join(palavras[i:i + tamanho_chunk]) for i in range(0, len(palavras), tamanho_chunk)]
    return chunks

with open(json_path, 'r', encoding='utf-8') as f:
    resolucoes = json.load(f)

for resolucao in resolucoes:
    titulo = resolucao["titulo"]
    pdf_path = os.path.join(pdf_folder, resolucao["arquivo"])
    
    texto_completo = ""
    with fitz.open(pdf_path) as pdf:
        for pagina in pdf:
            texto_completo += pagina.get_text()
    
    chunks = dividir_em_chunks(texto_completo)
    
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{titulo.replace(' ', '_').replace('/', '_')}_chunk_{i+1}.txt"
        chunk_path = os.path.join(output_folder, chunk_filename)
        with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)

print("Textos extraídos e salvos em chunks na pasta 'textos_extraidos'.")
