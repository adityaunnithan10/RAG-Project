import os 
from langchain_community.document_loaders import TextLoader,DirectoryLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


#document loader //
def load_documents(docs_path = 'docs') :

    if not os.path.exists(docs_path) :
        raise  FileNotFoundError(f'the given drectory at {docs_path} does not exist')

    loader = DirectoryLoader(
        path = docs_path,
        glob='*.txt',
        loader_cls=TextLoader,
        loader_kwargs= {'encoding' : 'UTF-8'}
    )

    documents =  loader.load()

    if len(documents) == 0 :
        raise FileNotFoundError(f'No text file found at {documents} path ')

    # for i,doc in enumerate(documents[:2]) :
    #     print(f"\nDocument {i+1}:")
    #     print(f'metadata : {doc.metadata}')
    #     print(f'content length :{len(doc.page_content)}')
    #     print(f'content preview :{doc.page_content[:100]}')
    #     print(f"source : {doc.metadata['source']}")

    return documents




#document chunking //
def split_document(documents,chunk_size = 800,chunk_overlap=0) :

    print("CHUNKING DOCUMENT")

    text_splitter = CharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap =chunk_overlap  
    )

    chunks  =  text_splitter.split_documents(documents)

    # if chunks :
    #     for i,chunk in enumerate(chunks[:5]) :
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)

    return chunks 



#create vectorstore and embed
def create_vectror_store(chunks,persist_directory="db/chroma_db"):
    print("Creating vector embedding and chroma db")

    embedding_model =  GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata= {"hnsw:space":"cosine"}
    )

    print("Vector Database Created")
    

def main() :

    print("running main")
    documents = load_documents()
    chunks = split_document(documents)
    create_vectror_store(chunks)

if __name__ == "__main__" :
    main()