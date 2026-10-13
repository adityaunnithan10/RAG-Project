from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage


load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model= "text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function= embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "Who owns tesla "

retriever = db.as_retriever(search_kwargs={"k":5})
relevant_documents =  retriever.invoke(query)



context = "" 
for doc in relevant_documents :
    context += f'-{doc.page_content}\n' #combine retrived documents



combined_input = f"""
Based on the following document, answer this question : {query}
Documents:
{context}
please provide clear, helpful answers only using the infromation from provided documents. If you cant answer the question using the documents , say "I dont have enough information to answer the question" 
"""
model =  ChatOpenAI(model= "gpt-4o")

messeges = [
    SystemMessage(content= "You are a helpful assitant"),
    HumanMessage(combined_input)
]

result =  model.invoke(messeges)
print(f'user query : {query}')
print(f'answer :\n{result.content}' )







