import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

if __name__ == "__main__":

    try:
        print("Hi...Im Ingesting")
        loader = TextLoader(
            "C:\\Users\\000154869\\Documents\\Curso ML\\langchainCourse\\langchain-medium-rag\\articule.txt",
            encoding="UTF-8",
        )
        document = loader.load()

        print("Splitting...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(f"created {len(texts)} chunks")

        embeddings = OllamaEmbeddings(model="llama3.2")

        print("ingesting...")
        PineconeVectorStore.from_documents(
            texts, embeddings, index_name=os.environ["INDEX_NAME"]
        )
        print("finish")
    except Exception as e:
        print(e)
        raise e
