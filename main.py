import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from pypdf import PdfReader
import glob

# === CONFIG ===
PDF_FOLDER = "./pdfs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === FastAPI app ===
app = FastAPI(title="RAGPsy Mental Health Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store PDF content
pdf_content = ""

def load_pdfs():
    global pdf_content
    content = []
    
    if os.path.exists(PDF_FOLDER):
        pdf_files = glob.glob(f"{PDF_FOLDER}/*.pdf")
        for pdf_file in pdf_files:
            try:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    content.append(page.extract_text())
            except Exception as e:
                print(f"Error reading {pdf_file}: {e}")
    
    pdf_content = "\n\n".join(content)
    print(f"✅ Loaded {len(content)} pages from PDFs")

def chat_with_groq(message):
    if not GROQ_API_KEY:
        return "❌ GROQ API key not configured"
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Simple context search - find relevant content
        context = ""
        if pdf_content:
            # Simple keyword matching
            keywords = message.lower().split()
            lines = pdf_content.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in keywords):
                    relevant_lines.append(line)
            
            context = "\n".join(relevant_lines[:5])  # Top 5 relevant lines
        
        prompt = f"""You are a compassionate mental health chatbot. Use the context below to answer the user's question kindly and helpfully.

Context from mental health resources:
{context}

User: {message}
Chatbot:"""

        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.status_code}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load PDFs on startup
@app.on_event("startup")
async def startup_event():
    load_pdfs()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_with_groq(request.question)
        return JSONResponse(content={"answer": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Mental Health Chatbot API is running."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
