import chromadb
import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma

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
        #texts = text_splitter.split_documents(document)
        doc1 = Document(page_content="foo is something")
        doc2 = Document(page_content="Peñarol peñaroool")
        texts = [doc1, doc2]
        print(f"created {len(texts)} chunks")

        embeddings = OllamaEmbeddings(model="llama3.2")

        print("creating DB...")
        #pc = Pinecone(pinecone_api_key=os.environ["PINECONE_API_KEY"])
        print("ingesting...")
        #index = pc.Index(os.environ["INDEX_NAME"])
        #vector_store = PineconeVectorStore(index, embeddings)
        #ids = []
        #for i in range (1, len(texts)):
        #    ids.append(i)
        #recods = asyncio.run( vector_store.aadd_documents(documents=texts, ids=ids) )
        #print(recods)
        
        #client = chromadb.HttpClient()
        print("intento desde documents...")
        Chroma(
            embedding_function=embeddings,
            create_collection_if_not_exists=True,
            collection_name="prueba",
            persist_directory="./chroma_langchain_db"
        )
        Chroma.from_documents(documents=texts, embedding=embeddings)
        print("finish")
    except Exception as e:
        print(e)
        raise e
