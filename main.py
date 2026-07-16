import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# This is it loading.


load_dotenv()


# Feel Free to change the placeholders with your choice of model

model = ChatOpenAI(model="YOUR_MODEL_NAME", temperature=0.2)

embeddings = OpenAIEmbeddings(model="YOUR_EMBEDDING_NAME")

# MAKEING HTE RAG YEAAAH

print("LOADING(IDK WHY I USED PRINT HERE)")

loader = TextLoader("sample.txt")

docs = loader.load()

# Makes a Vector DataBase

textsplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

splits = textsplitter.split_documents(docs)

vectordatastore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectordatastore.as_retriever(search_kwargs={"k": 3})

# Making sure it doesn't have A D1 crashout when you use he not Dan Smith

systemprompt = (
    "In the chat history with the Client question "
    "Could use context in the chat history, "
    "just make a question which can be understood "
    "without the chat history. DO NOT ANSWER THE QUESTION "
    "Just remake the question if you need to. otherwise just return it AS IS"
)

# Turns out it is bad to name it Make_proompt context who knew?


contextualizeQprompt = ChatPromptTemplate.from_messages([
    ("system", systemprompt),  # Fixed: linked to systemprompt variable
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# lowkey I had to use AI to make a the text because it was not getting me


Qasystemprompt = (
    "You are a helpful assistant. Use the following pieces of context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "Context:\n{context}"
)

Qaprompt = ChatPromptTemplate.from_messages([
    ("system", Qasystemprompt),  # Fixed: fixed typo from Aasystemprompt
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# connects it


questionToAnswerchain = create_stuff_documents_chain(model, Qaprompt)  # Fixed: linked to Qaprompt

# makes the RAG

historyawareretriever = create_history_aware_retriever(model, retriever, contextualizeQprompt)


RAGchain = create_retrieval_chain(historyawareretriever, questionToAnswerchain)


# makes a ses mem

sessionmemories = {}

# Moved function up here cuz AI told mw I am a dumb dumb and it needs to be up


def getsessionhistory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in sessionmemories:
        sessionmemories[session_id] = InMemoryChatMessageHistory()
    return sessionmemories[session_id]

# making it work


conversationalRAGchain = RunnableWithMessageHistory(
    RAGchain,
    getsessionhistory,
    input_messages_key="input",        
    history_messages_key="chat_history", 
    output_messages_key="answer",
)

# It is configing it


config = {"configurable": {"session_id": "RAG-CHATBOT-session"}}

# YAYAYAY We can START!

print("Chatbot is WORKING Type 'exit' to stop.\n")


while True:
    userinput = input("User: ")
    
    # Making people Leave
  
    if userinput.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

  
        
    print("AI-CHATBOT: ", end="", flush=True)
    
    # The response
  
    response = conversationalRAGchain.invoke({"input": userinput}, config=config)

  
    print(response["answer"], "\n")
