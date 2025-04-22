# APLICAÇÃO DE MODELOS DE LINGUAGEM (LLM) PARA OTIMIZAÇÃO DA BUSCA DE LEGISLAÇÕES DO PPGCAP

## 📌 Subtítulo
**EMBEDDINGS SEMÂNTICOS COM O MODELO PARAPHRASE-MULTILINGUAL-MINILM-L12-V2**

---

## 📚 Descrição do Projeto

Este sistema utiliza **modelos de linguagem natural (LLMs)** e **indexação vetorial com FAISS** para permitir **busca inteligente** em um acervo de portarias, resoluções e documentos administrativos da Universidade do Estado de Santa Catarina (UDESC).

Combinando **embeddings semânticos** e **busca literal**, o sistema identifica documentos mais relevantes com base na **semelhança textual**, retornando links diretos para os PDFs hospedados no site da UDESC.

---

## 🧰 Tecnologias Utilizadas

- 🐍 Python 3
- 🤖 [SentenceTransformers](https://www.sbert.net/) (modelo `paraphrase-multilingual-MiniLM-L12-v2`)
- 📊 [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- 📄 PyMuPDF (`fitz`) para extração de texto de PDFs
- 📦 Unidecode, NumPy, Pickle, JSON

---

## 📁 Estrutura dos Arquivos

```bash
SI_IA_UDESC/
├── busca_semantica_udesc.ipynb       # Notebook principal com o sistema de busca
├── index_faiss.idx                   # Índice vetorial FAISS
├── nomes_textos.pkl                  # Nomes dos arquivos indexados
├── mapeamento_links.json            # Mapeamento de arquivos .txt para URLs dos PDFs
├── textos_extraidos/                # Arquivos .txt extraídos dos PDFs
├── portarias/                       # PDFs originais


🔍 Como Usar
Execute o notebook busca_semantica_udesc.ipynb
Digite sua pergunta no prompt
O sistema fará:
Busca literal no texto completo
Caso não encontre, aplica busca semântica com FAISS
Exibe:
Nome do arquivo
Trecho mais relevante
Link direto para o PDF original


🧪 Exemplo de Consulta
Entrada: programa de intercâmbio
Saída: 📄 Resolução_0132014_-_CONSEPE
🧾 Trecho relevante: § 2º A incorporação dos alunos no programa de intercâmbio sujeitar-se-á às regras
🔗 [Clique para abrir o PDF original](https://www.udesc.br/...)


👨‍🏫 Informações Acadêmicas
Disciplina: Sistemas Inteligentes
Professor: Dr. Rafael Stubs Parpinelli
Curso: Doutorado Acadêmico em Computação Aplicada
Universidade: Universidade do Estado de Santa Catarina – UDESC

📄 Licença
Este projeto é de uso acadêmico e experimental.
