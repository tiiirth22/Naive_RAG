import os
import json
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from scrape_wikipedia import load_scraped_data


class VectorDatabase:
    """Manages ChromaDB vector database operations."""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to ChromaDB directory
            embedding_model: HuggingFace model for embeddings
        """
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = None
    
    def create_collection(self, collection_name: str = "rag_documents", reset: bool = False):
        """
        Create or get ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            reset: Whether to delete existing collection
        """
        if reset and collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=collection_name)
            print(f"Reset collection: {collection_name}")
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Using collection: {collection_name}")
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start += chunk_size - overlap
        
        return chunks
    
    def add_documents(self, reset: bool = False) -> int:
        """
        Add all scraped Wikipedia documents to vector database.
        
        Args:
            reset: Whether to reset the collection
            
        Returns:
            Number of chunks added
        """
        self.create_collection(reset=reset)
        
        scraped_data = load_scraped_data()
        
        if not scraped_data:
            print(f"No scraped data found in {DATA_PATH}")
            return 0
        
        chunk_count = 0
        
        for keyword, articles in scraped_data.items():
            print(f"\nProcessing keyword: {keyword}")
            
            for article_idx, article in enumerate(articles):
                title = article.get('title', 'Unknown')
                url = article.get('url', '')
                content = article.get('content', '')
                
                if not content:
                    print(f"  ⚠ Empty content for article: {title}")
                    continue
                
                # Chunk the article
                chunks = self.chunk_text(content)
                print(f"  Article '{title}' -> {len(chunks)} chunks")
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{keyword}_{article_idx}_{chunk_idx}"
                    
                    # Generate embedding
                    try:
                        embedding = self.embedding_model.encode(chunk_text, convert_to_tensor=True).tolist()
                    except Exception as e:
                        print(f"    ✗ Error embedding chunk {chunk_id}: {str(e)}")
                        continue
                    
                    # Prepare metadata
                    metadata = {
                        "article_title": title,
                        "source_url": url,
                        "chunk_id": chunk_id,
                        "keyword": keyword,
                        "chunk_index": chunk_idx
                    }
                    
                    # Add to collection
                    try:
                        self.collection.add(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            documents=[chunk_text],
                            metadatas=[metadata]
                        )
                        chunk_count += 1
                    except Exception as e:
                        print(f"    ✗ Error adding chunk {chunk_id}: {str(e)}")
        
        print(f"\n✓ Added {chunk_count} chunks to vector database")
        return chunk_count
    
    def get_collection_size(self) -> int:
        """
        Get number of documents in collection.
        
        Returns:
            Number of documents
        """
        if not self.collection:
            return 0
        return self.collection.count()


def initialize_vector_db(reset: bool = False) -> VectorDatabase:
    """
    Initialize and populate vector database.
    
    Args:
        reset: Whether to reset existing database
        
    Returns:
        VectorDatabase instance
    """
    db = VectorDatabase()
    db.add_documents(reset=reset)
    return db


if __name__ == "__main__":
    print("Initializing vector database...\n")
    db = initialize_vector_db(reset=True)
    print(f"\n✓ Vector database initialized")
    print(f"✓ Total chunks in database: {db.get_collection_size()}")