import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# === CONFIG ===
PDF_FOLDER = "./pdfs"
DB_PATH = "./simple_db.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-70b-8192"

# === FastAPI app ===
app = FastAPI(title="RAGPsy Mental Health Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global variables ===
embeddings_model = None
documents = []
embeddings = []
llm = None

# === 1. Initialize LLM ===
def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("⚠️ Please set your Groq API key in environment variables.")
    
    return ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL
    )

# === 2. Load and process documents ===
def load_documents():
    print("📄 Loading PDFs...")
    loader = DirectoryLoader(PDF_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    return chunks

# === 3. Create embeddings ===
def create_embeddings(docs):
    print("🔍 Creating embeddings...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)
    
    # Save to pickle file
    with open(DB_PATH, 'wb') as f:
        pickle.dump({'texts': texts, 'embeddings': embeddings, 'docs': docs}, f)
    
    return texts, embeddings, docs

# === 4. Load existing embeddings ===
def load_embeddings():
    print("📂 Loading existing embeddings...")
    with open(DB_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['texts'], data['embeddings'], data['docs']

# === 5. Search function ===
def search_documents(query, top_k=3):
    global embeddings_model, documents, embeddings
    
    query_embedding = embeddings_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    context = []
    for idx in top_indices:
        context.append(documents[idx].page_content)
    
    return "\n\n".join(context)

# === 6. Chat function ===
def chat_with_bot(question):
    try:
        context = search_documents(question)
        
        prompt = f"""You are a compassionate mental health chatbot. Use the context to answer the user's question kindly.

Context:
{context}

User: {question}
Chatbot:"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === 7. Initialize system ===
def initialize_system():
    global embeddings_model, documents, embeddings, llm
    
    try:
        llm = initialize_llm()
        embeddings_model = SentenceTransformer(EMBED_MODEL)
        
        if os.path.exists(DB_PATH):
            texts, emb, docs = load_embeddings()
            documents = docs
            embeddings = emb
        else:
            docs = load_documents()
            texts, emb, docs = create_embeddings(docs)
            documents = docs
            embeddings = emb
            
        print("✅ System initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return False

# === 8. API Endpoints ===
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_with_bot(request.question)
        return JSONResponse(content={"answer": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Mental Health Chatbot API is running."}

# === 9. Startup ===
@app.on_event("startup")
async def startup_event():
    initialize_system()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
