from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_CHUNKS


class DocumentRetriever:
    """Retrieves relevant documents from vector database."""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize document retriever.
        
        Args:
            db_path: Path to ChromaDB directory
            embedding_model: HuggingFace model for embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS
    ) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        retrieved_chunks = []
        
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity_score = 1 - distance
                
                chunk = {
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for idx, chunk in enumerate(retrieved_chunks, 1):
            article_title = chunk["metadata"].get("article_title", "Unknown")
            similarity = chunk["similarity_score"]
            content = chunk["content"]
            
            part = f"[Source {idx}: {article_title} (Similarity: {similarity:.2%})]"
            part += f"\n{content}\n"
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def get_sources(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Extract unique sources from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            List of unique sources with metadata
        """
        sources = {}
        
        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            url = metadata.get("source_url", "")
            title = metadata.get("article_title", "Unknown")
            similarity = chunk["similarity_score"]
            
            if url not in sources:
                sources[url] = {
                    "title": title,
                    "url": url,
                    "max_similarity": similarity
                }
            else:
                # Keep track of maximum similarity score
                sources[url]["max_similarity"] = max(
                    sources[url]["max_similarity"],
                    similarity
                )
        
        return list(sources.values())


if __name__ == "__main__":
    retriever = DocumentRetriever()
    
    # Test retrieval
    test_query = "What is Python?"
    print(f"Query: {test_query}\n")
    
    results = retriever.retrieve(test_query)
    
    print(f"Retrieved {len(results)} chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"[Chunk {i}] Similarity: {chunk['similarity_score']:.2%}")
        print(f"Source: {chunk['metadata']['article_title']}")
        print(f"Content: {chunk['content'][:200]}...\n")
    
    context = retriever.format_context(results)
    print("\nFormatted Context:")
    print(context)