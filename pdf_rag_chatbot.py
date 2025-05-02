import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import glob
import shutil
import tempfile

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Hugging Face API token from environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-hugging-face-api-token"  # Replace with your token or set as env variable

# Global variables to store vector store and QA chain
vector_store = None
qa_chain = None

# Step 1: Load and process PDFs
def load_pdfs(pdf_folder):
    documents = []
    for pdf_file in glob.glob(f"{pdf_folder}/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents

# Step 2: Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# Step 3: Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

# Step 4: Initialize the RAG chain
def initialize_rag_chain(vector_store):
    # Use Hugging Face's Inference API for LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Lightweight model for free tier
        model_kwargs={"max_length": 500, "temperature": 0.7}
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """You are a helpful assistant tasked with answering questions based on the provided PDF documents. Use the following context to provide a clear, concise, and accurate answer. If the context does not contain enough information to answer the question, state that explicitly.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

# API endpoint to upload PDFs and initialize the chatbot
@app.post("/upload_pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global vector_store, qa_chain
    pdf_folder = "./pdfs"
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Save uploaded PDFs
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        file_path = os.path.join(pdf_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
    
    # Process PDFs
    try:
        documents = load_pdfs(pdf_folder)
        if not documents:
            raise HTTPException(status_code=400, detail="No PDFs found")
        
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)
        qa_chain = initialize_rag_chain(vector_store)
        return {"message": "PDFs processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for query request
class QueryRequest(BaseModel):
    question: str

# API endpoint to query the chatbot
@app.post("/query")
async def query(request: QueryRequest):
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Chatbot not initialized. Upload PDFs first.")
    try:
        response = qa_chain.run(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
