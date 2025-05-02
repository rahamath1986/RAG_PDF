import os
import logging
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your-hugging-face-api-token")

# Global variables
vector_store = None
qa_chain = None

def load_pdfs(pdf_folder):
    logger.info("Loading PDFs from %s", pdf_folder)
    documents = []
    for pdf_file in glob.glob(f"{pdf_folder}/*.pdf"):
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            logger.error("Error loading PDF %s: %s", pdf_file, str(e))
    return documents

def split_documents(documents):
    logger.info("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced for lower memory usage
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    logger.info("Creating vector store")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}  # Optimize for efficiency
    )
    return vector_store

def initialize_rag_chain(vector_store):
    logger.info("Initializing RAG chain")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
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

@app.post("/upload_pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global vector_store, qa_chain
    pdf_folder = "./pdfs"
    os.makedirs(pdf_folder, exist_ok=True)
    
    logger.info("Received %d files for upload", len(files))
    for file in files:
        if not file.filename.endswith(".pdf"):
            logger.warning("Invalid file type: %s", file.filename)
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        file_path = os.path.join(pdf_folder, file.filename)
        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logger.info("Saved file: %s", file_path)
        except Exception as e:
            logger.error("Error saving file %s: %s", file.filename, str(e))
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    try:
        documents = load_pdfs(pdf_folder)
        if not documents:
            logger.warning("No PDFs found in folder")
            raise HTTPException(status_code=400, detail="No PDFs found")
        
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)
        qa_chain = initialize_rag_chain(vector_store)
        logger.info("PDFs processed successfully")
        return {"message": "PDFs processed successfully"}
    except Exception as e:
        logger.error("Error processing PDFs: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(request: QueryRequest):
    global qa_chain
    if qa_chain is None:
        logger.warning("Query attempted before chatbot initialization")
        raise HTTPException(status_code=400, detail="Chatbot not initialized. Upload PDFs first.")
    try:
        logger.info("Processing query: %s", request.question)
        response = qa_chain.run(request.question)
        logger.info("Query response: %s", response)
        return {"answer": response}
    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)