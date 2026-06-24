import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval Configuration
TOP_K_CHUNKS = 5

# Ensure directories exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Validation
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file.")