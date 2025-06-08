import requests
from bs4 import BeautifulSoup
import os
import json

# URL da página de legislações
url = "https://www.udesc.br/cct/ppgca/legislacoes"

# Lista de resoluções internas do PPGCAP
resolucoes = [
    "Resolução 047/2025 - Critérios disciplinas de Aceitação de Artigo/ACA",
    "Resolução 046/2024 - Validação de disciplina no plano de curso",
    "Resolucao 042/2023 - Distribuicao_Bolsas alterada pela Resolucão 044/2024",
    "Resolução 041/2023 - Regime Didático PPGCAP",
    "Resolução 040/2023 - Credenciamento, recredenciamento, descredenciamento e acompanhamento do corpo docente",
    "Resolução 039/2023 - Número de orientações",
    "Resolução 038/2023 - Produtividade Docente",
    "Resolução 036/2021 - Seminários de Inovação em Computação",
    "Resolução 032/2020 - Critérios desligamento de alunos",
    "Resolução 031/2020 - Requisitos para defesa",
    "Resolução 030/2020 - Prazos e aulas não presenciais durante a pandemia",
    "Resolução 025/2018 - Qualificação",
    "Resolução 024/2018 - Exame de Proficiência Alterada pela Resolução 045/2024",
    "Resolução 019/2015 - Validação de Disciplina - Alterada pela Resolução 020/2016 e Resolucão 043/2024",
    "Resolução 007/2012 - Docência Orientada (Anexo 1) (Anexo 2)"
]

# Pasta para salvar os PDFs
pasta_pdfs = "C:/Users/dayax/Downloads/PDFs_Udesc"
os.makedirs(pasta_pdfs, exist_ok=True)

# Lista para armazenar os metadados das resoluções
metadados_resolucoes = []

# Função para baixar um PDF
def baixar_pdf(titulo, link):
    response = requests.get(link)
    nome_arquivo = f"{titulo.replace(' ', '_').replace('/', '_')}.pdf"
    caminho_arquivo = os.path.join(pasta_pdfs, nome_arquivo)
    with open(caminho_arquivo, 'wb') as f:
        f.write(response.content)
    return nome_arquivo

# Acessar a página e buscar os links dos PDFs
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Procurar os links das resoluções específicas
for resolucao in resolucoes:
    link_tag = soup.find('a', text=resolucao)
    if link_tag:
        link_pdf = link_tag['href']
        nome_arquivo = baixar_pdf(resolucao, link_pdf)
        metadados_resolucoes.append({
            "titulo": resolucao,
            "arquivo": nome_arquivo,
            "link": link_pdf
        })

# Salvar os metadados em um arquivo JSON
caminho_json = os.path.join(pasta_pdfs, "resolucoes_ppgcap.json")
with open(caminho_json, 'w', encoding='utf-8') as f:
    json.dump(metadados_resolucoes, f, ensure_ascii=False, indent=4)

print(f"PDFs baixados e metadados salvos em {caminho_json}.")
