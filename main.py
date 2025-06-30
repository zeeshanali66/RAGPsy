import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# === CONFIG ===
PDF_FOLDER = "./pdfs"  # Folder where PDF files are stored
DB_PATH = "./chroma_db"  # Path to store vector database
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model for text
GROQ_MODEL = "llama3-70b-8192"  # The Groq model you're using

# === FastAPI app ===
app = FastAPI()

# === 1. Initialize LLM (Language Model) ===
def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY") or "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" # Fetching API key from environment variable
    if not groq_api_key:
        raise ValueError("⚠️ Please set your Groq API key in the environment variable.")
    
    # Initialize Groq LLM with provided API key
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,  # API key from environment
        model_name=GROQ_MODEL  # Model to use
    )
    return llm


# === 2. Create Vector DB from PDFs ===
def create_vector_db():
    print("📄 Loading and chunking PDFs...")
    loader = DirectoryLoader(PDF_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print("🔍 Creating embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    print("💾 Saving to Chroma vector DB...")
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    vector_db.persist()
    return vector_db


# === 3. Setup QA Chain ===
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()

    # Define a custom prompt template for the chatbot
    prompt_template = """
    You are a compassionate mental health chatbot. Use the context to answer the user's question kindly.

    Context:
    {context}

    User: {question}
    Chatbot:"""

    # Create a prompt template for the system
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Setup the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


# === 4. Initialize LLM and Vector DB ===
llm = initialize_llm()
if not os.path.exists(DB_PATH):
    vector_db = create_vector_db()
else:
    print("📂 Loading existing Chroma vector DB...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)


# === 5. FastAPI Endpoint ===
class ChatRequest(BaseModel):
    question: str  # User's question

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Fetching response from the QA chain
        response = qa_chain.run(request.question)
        return JSONResponse(content={"answer": response})
    except Exception as e:
        # If there's an error, return a detailed message
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
def root():
    return {"message": "Mental Health Chatbot API is running."}


# To run the application locally:
# uvicorn app:app --reload
