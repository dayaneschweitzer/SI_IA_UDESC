from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

import os

# Caminho base dos PDFs
pdf_base_path = os.path.abspath("documentos")

# Carregar PDFs com metadados
loader = PyPDFLoader("documentos/*.pdf")
documents = loader.load()
for doc in documents:
    # Adiciona o caminho completo como metadado
    doc.metadata["source"] = os.path.abspath(doc.metadata.get("source", doc.metadata.get("file_path", "desconhecido")))

# Criar embeddings e indexar
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents, embedding)

# Configurar o modelo
llm = OllamaLLM(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)

# Loop para receber perguntas do usuário
while True:
    query = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
    if query.lower() in ["sair", "exit", "quit"]:
        break

    resposta = qa.invoke(query)

    print("\n Resposta:")
    print(resposta["result"])

    print("\n📄 Fonte(s):")
    for doc in resposta["source_documents"]:
        source_path = doc.metadata.get("source", "desconhecido")
        print(f"- {os.path.basename(source_path)}")
        print(f"  Link: file:///{source_path.replace(os.sep, '/')}")
