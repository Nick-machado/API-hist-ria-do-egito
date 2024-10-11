import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter

def database():
    load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env

    # Obtém o diretório atual do arquivo
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Lista todos os arquivos no diretório 'rag'
    files = os.listdir(os.path.join(current_dir, "rag"))

    # Define o diretório persistente para o banco de dados Chroma
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    # Verifica se o diretório persistente já existe
    if not os.path.exists(persistent_directory):
        print("\n ---Persistent directory does not exist. Initializing vector store---\n")

        # Itera sobre cada arquivo no diretório 'rag'
        for file in files:
            file_path = os.path.join(current_dir, "rag", file)  # Caminho completo do arquivo
            loader = PyPDFLoader(file_path)  # Carrega o PDF usando o PyPDFLoader
            documents = loader.load_and_split()  # Carrega e divide o documento em páginas

            # Configura o divisor de texto para criar chunks dos documentos
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)  # Divide os documentos em chunks

            print("\n--- Document chunks information ---")
            print(f"Number of document chunks: {len(docs)}")  # Número total de chunks
            print(f"Sample chunk:\n{docs[0].page_content}\n")  # Exemplo de um chunk

            print("\n--- Creating embeddings ---")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Cria embeddings dos chunks
            print("\n--- Finished creating embeddings ---")

            print("\n--- Creating vector store ---")
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_directory
            )  # Cria o banco de dados vetorial e salva no diretório persistente
            print('\n--- Finished creating vector store ---')
    else:
        print("\n ---Persistent directory already exist.---\n")

# Executa a função database
database()
