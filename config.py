import os
from dotenv import load_dotenv

load_dotenv(override=True)

CHROMADB_PATH = "chroma_db"
DOCS_PATH = "docs"
EMBEDDINGS_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")