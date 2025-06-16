# APLICAÇÃO DE MODELOS DE LINGUAGEM (LLM) PARA OTIMIZAÇÃO DA BUSCA DE LEGISLAÇÕES DO PPGCAP

## Subtítulo  
***Busca Inteligente com Modelo RAG baseado em Título + Texto Completo***

---

## Descrição Geral do Projeto

Este projeto desenvolve um sistema de busca semântica e resposta automatizada para documentos legislativos do Programa de Pós-Graduação em Computação Aplicada (PPGCAP) da UDESC. A aplicação utiliza um modelo de linguagem LLM (via Ollama + LangChain) para identificar o documento mais relevante a partir de uma lista de resoluções disponíveis, e em seguida, responde à pergunta do usuário com base no texto completo desse documento.

A relevância do documento é determinada diretamente por um modelo LLM por meio de raciocínio textual.

---

## Tecnologias e Bibliotecas

Python 3.x

Flask – Interface Web (chat)
PyMuPDF (fitz) – Extração de texto dos PDFs
Sentence Transformers – Geração de embeddings (caso desejado para uso posterior)
LangChain e LangChain-Ollama – Interação com LLM
Unidecode – Normalização textual
JSON, Pickle – Armazenamento de dados


## Estrutura de Diretórios
```
SI_IA_UDESC/
├── PDFs_Udesc/                  # PDFs das resoluções originais
├── textos_extraidos/           # Textos extraídos dos PDFs
├── vetores/                    # Dados estruturados: full_textos.pkl, metadados.pkl
├── app.py                      # Aplicação principal com LLM e RAG baseado em títulos
├── gerar_embeddings.py         # Script legado (não mais utilizado na versão RAG)
├── gerar_pdf.py                # Script para extrair textos dos PDFs
├── index.html                  # Interface de chat web
├── README.md                   # Este documento
├── requirements.txt            # Bibliotecas necessárias

```

## Instruções de Execução

Certifique-se de que os PDFs estão em PDFs_Udesc/.

Gere os textos com o script:
python gerar_textos.py

Inicie a aplicação web:
python app.py

Acesse via navegador:
http://127.0.0.1:5000

**Como funciona:**
O usuário digita uma pergunta no chat.
O sistema apresenta a pergunta ao LLM junto com uma lista de títulos de resoluções.
O LLM escolhe a mais relevante.
O conteúdo do documento é usado para gerar uma resposta clara e objetiva.
O título do documento é apresentado como hiperlink clicável para o PDF.

Disciplina: Sistemas Inteligentes
Professor: Dr. Rafael Stubs Parpinelli
Programa: Doutorado Acadêmico em Computação Aplicada
Instituição: Universidade do Estado de Santa Catarina – UDESC

## Licença
Este projeto é de uso acadêmico, restrito a fins de pesquisa e experimentação dentro do escopo da disciplina de Sistemas Inteligentes.