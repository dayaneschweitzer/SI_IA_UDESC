Busca Semântica de Legislação PPGCAP
Busca Semântica de Legislação PPGCAP
Este projeto realiza busca semântica e literal sobre textos legais relacionados ao Programa de Pós-Graduação em Computação Aplicada (PPGCAP).
Utiliza embeddings com Sentence Transformers, redução de dimensionalidade via PCA e FAISS para consultas eficientes.

Estrutura do projeto:
.
├── benchmark/
│   ├── benchmark_busca.py
│   └── plot_benchmark.py
├── buscar.py
├── extrator.py
├── gerar_embeddings.py
├── main.py
├── mapeamento_links.json
├── portarias/
├── textos_extraidos/
└── requirements.txt

1. Instalação de dependências

pip install -r requirements.txt

Exemplo de pacotes necessários:

sentence-transformers
faiss-cpu
scikit-learn
pandas
numpy
matplotlib
seaborn
unidecode

2. Preparação dos dados
Coloque os arquivos PDF na pasta portarias/.

Ajuste os arquivos de mapeamento (mapeamento_links.json) conforme necessidade.

3. Como rodar

Executar o sistema:

python main.py

Ao executar, o menu será exibido:

== Projeto: Busca Semântica de Legislação PPGCAP ==

Escolha a opção:
1. Executar busca interativa
2. Executar benchmark e gerar gráficos
3. Executar pipeline completa (extração + embeddings)
4. Funcionalidades

1. Executar busca interativa
Escolha um modelo:
paraphrase-multilingual-MiniLM-L12-v2 ou all-mpnet-base-v2.

Digite sua pergunta livremente.

O sistema retorna resultados via busca semântica.

Se não encontrar, faz busca literal automaticamente.

2. Executar benchmark e gerar gráficos

Executa benchmark_busca.py: avalia os modelos com diversas perguntas, top_k e limiar.
Gera gráficos comparativos com plot_benchmark.py.

Gráficos gerados:

Heatmap por modelo

Comparativo: modelo x limiar x top_k

Os arquivos CSV de benchmark e imagens de gráficos são salvos automaticamente.

3. Executar pipeline completa

Realiza:

Extração dos textos dos PDFs.
Divisão em chunks.

4. Geração de embeddings para todos os modelos.
Aplicação de PCA para redução dimensional.
Construção e salvamento de índices FAISS.
Arquivos gerados por modelo:
embeddings_<modelo>.pkl
pca_model_<modelo>.pkl
index_faiss_<modelo>.idx
nomes_textos_<modelo>.pkl

5. Configurações importantes
MODELOS: configurados em main.py e gerar_embeddings.py.

Dimensão PCA: padrão 256. Pode ser ajustado em gerar_embeddings.py.

Diretório de textos: textos_extraidos.

6. Observações
O sistema é modular: facilmente adaptável para novos modelos ou datasets.

Busca eficiente graças ao uso de FAISS e PCA.

Benchmark automatizado: mede desempenho dos modelos de forma rápida.

7. Exemplo de uso rápido:

python main.py

Escolha a opção:
1. Executar busca interativa
2. Executar benchmark e gerar gráficos
3. Executar pipeline completa (extração + embeddings)
→ Escolha 1 para busca interativa.
→ Escolha 2 para gerar benchmarks e gráficos.
→ Escolha 3 para preparar os embeddings e índices.

8. Contato
Para dúvidas ou sugestões:
Desenvolvido por Dda. Dayane da Silva Xavier Schweitzer