import os 
from langchain_community.document_loaders import TextLoader,DirectoryLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

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

    for i,doc in enumerate(documents[:2]) :
        print(f"\nDocument {i+1}:")
        print(f'metadata : {doc.metadata}')
        print(f'content length :{len(doc.page_content)}')
        print(f'content preview :{doc.page_content[:100]}')
        print(f"source : {doc.metadata['source']}")


def main() :
    print("Main Function")
    load_documents()

if __name__ == "__main__" :
    main()