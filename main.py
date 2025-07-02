import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import glob

# === CONFIG ===
PDF_FOLDER = "./pdfs"
GROQ_MODEL = "llama3-70b-8192"

# === FastAPI app ===
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store PDF content
pdf_content = ""

# === 1. Initialize system ===
def load_pdfs():
    global pdf_content
    content = []
    
    if os.path.exists(PDF_FOLDER):
        # Simple text file reading instead of PDF parsing
        txt_files = glob.glob(f"{PDF_FOLDER}/*.txt")
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as file:
                    content.append(file.read())
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
    
    pdf_content = "\n\n".join(content)
    print(f"✅ Loaded content from {len(content)} files")

# === 2. Chat with Groq ===
def chat_with_groq(question):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "❌ GROQ API key not configured"
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple context search
        context = ""
        if pdf_content:
            keywords = question.lower().split()
            lines = pdf_content.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in keywords if len(keyword) > 2):
                    relevant_lines.append(line)
            
            context = "\n".join(relevant_lines[:10])  # Top 10 relevant lines
        
        prompt_template = f"""
        You are a compassionate mental health chatbot. Use the context to answer the user's question kindly.

        Context:
        {context}

        User: {question}
        Chatbot:"""

        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt_template}],
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

# Load content on startup
load_pdfs()

# === 3. FastAPI Endpoint ===
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

# === 4. Run the app ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
