# APLICAÇÃO DE MODELOS DE LINGUAGEM (LLM) PARA OTIMIZAÇÃO DA BUSCA DE LEGISLAÇÕES DO PPGCAP
Subtítulo: Embeddings Semânticos com o Modelo paraphrase-multilingual-MiniLM-L12-v2

Descrição Geral do Projeto:
Este projeto implementa um sistema de busca semântica inteligente aplicado a documentos administrativos e legislativos do Programa de Pós-Graduação em Computação Aplicada (PPGCAP) da UDESC.

Utiliza-se um modelo pré-treinado de transformador semântico multilíngue para gerar embeddings vetoriais de segmentos de texto extraídos de arquivos PDF. A busca é realizada por similaridade vetorial utilizando a biblioteca FAISS, permitindo recuperar documentos com base no significado da consulta, e não apenas por correspondência literal.

Tecnologias e Bibliotecas
Python 3.x
Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
FAISS (Facebook AI Similarity Search)
PyMuPDF (fitz) – Extração de texto de PDFs
Unidecode – Normalização textual
NumPy – Operações vetoriais
Pickle e JSON – Armazenamento de dados intermediários

Estrutura de Diretórios
`SI_IA_PPGCAP/
├── textos_extraidos/           # Segmentos textuais (chunks)
├── portarias/                  # Arquivos PDF originais
├── extrator.py                 # Módulo de extração de texto e segmentação em chunks
├── gerar_embeddings.py         # Geração e indexação de embeddings com FAISS
├── buscar.py                   # Funções de busca semântica e recuperação de trechos
├── mapeamento_links.json       # Mapeamento de nomes de arquivos para URLs
├── nomes_textos.pkl            # Lista de nomes indexados
├── index_faiss.idx             # Índice vetorial salvo pelo FAISS
├── main.py                     # Executável principal (pipeline completo)`

Instruções de Execução
1. Certifique-se de que os arquivos PDF estão na pasta portarias/.
2. Execute o arquivo main.py com Python 3:

`python main.py`

O sistema irá:
--> Extrair e segmentar o texto dos documentos.
--> Gerar embeddings e criar o índice FAISS.
--> Iniciar o modo interativo para consultas.
--> Digite uma pergunta ou expressão no terminal para buscar documentos relevantes.

A aplicação retorna:
--> Nome do arquivo relevante
--> Trecho com alta similaridade semântica
--> Link direto para o PDF original

Exemplo de Consulta
Entrada:
`programa de intercâmbio`

Saída:
`Arquivo: Resolução_0132014_-_CONSEPE
Trecho: “§ 2º A incorporação dos alunos no programa de intercâmbio sujeitar-se-á às regras [...]”
Link: Disponibilizado conforme o mapeamento mapeamento_links.json`

Informações Acadêmicas
Disciplina: Sistemas Inteligentes
Professor: Dr. Rafael Stubs Parpinelli
Programa: Doutorado Acadêmico em Computação Aplicada
Instituição: Universidade do Estado de Santa Catarina – UDESC

Licença
Este projeto é de uso acadêmico, restrito a fins de pesquisa e experimentação dentro do escopo da disciplina de Sistemas Inteligentes.


