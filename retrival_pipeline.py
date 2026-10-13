from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


#retriever 
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "In What year did tesla begin producation"

retriever = db.as_retriever(search_kwargs={"k":5})

relevant_documents = retriever.invoke(query)

print(f"user query : {query}")

for i, doc in enumerate(relevant_documents, 1):
    print(f"Document {i}:\n{doc.page_content}\n")