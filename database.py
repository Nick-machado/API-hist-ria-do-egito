import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter

def database():

    load_dotenv()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(os.path.join(current_dir, "rag"))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    if not os.path.exists(persistent_directory):
        print("\n ---Persistent directory does not exist. Initializing vector store---\n")

        for file in files:
            file_path = os.path.join(current_dir, "rag", file)
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            print("\n--- Document chunks information ---")
            print(f"Number of document chunks: {len(docs)}")
            print(f"Sample chunk:\n{docs[0].page_content}\n")

            print("\n--- Creating embeddings ---")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            print("\n--- Finished creating embeddings ---")

            print("\n--- Creating vector store ---")
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_directory
            )
            print('\n--- Finished creating vector store ---')
    else:
        print("\n ---Persistent directory already exist.---\n")

database()