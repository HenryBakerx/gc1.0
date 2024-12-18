# ingest.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pickle
import os

load_dotenv()  # Make sure to call this before os.getenv()

PDF_PATH = "/Users/chiel/Documents/bronnen/UAV 2012.pdf"  # Path to your local PDF

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's set in .env or as an environment variable.")

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text
raw_text = get_pdf_text(PDF_PATH)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
text_chunks = text_splitter.split_text(raw_text)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Save FAISS index to disk
vectorstore.save_local("faiss_index")

# Optionally, store metadata in a pickle file if needed
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("Vectorstore created and saved successfully.")
