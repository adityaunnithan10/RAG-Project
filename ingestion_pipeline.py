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

def main() :
    print("Main Function")

if __name__ == "__main__" :
    main()