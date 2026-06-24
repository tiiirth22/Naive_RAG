from typing import List, Tuple, Dict
from langchain.chat_models import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from config import GROQ_API_KEY, MODEL_NAME
from retrieve import DocumentRetriever
from query_rewriter import QueryRewriter


class RAGChat:
    """Main RAG chatbot class combining retrieval, rewriting, and generation."""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        api_key: str = GROQ_API_KEY
    ):
        """
        Initialize RAG chat.
        
        Args:
            model_name: Name of the Groq model
            api_key: Groq API key
        """
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.7
        )
        self.retriever = DocumentRetriever()
        self.query_rewriter = QueryRewriter(model_name, api_key)
        self.conversation_history: List[Tuple[str, str]] = []
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite user query using conversation history.
        
        Args:
            query: User query
            
        Returns:
            Rewritten query
        """
        if not self.conversation_history:
            return query
        
        return self.query_rewriter.rewrite_query(query, self.conversation_history)
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """
        Retrieve relevant context for query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (retrieved_chunks, formatted_context)
        """
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        formatted_context = self.retriever.format_context(retrieved_chunks)
        
        return retrieved_chunks, formatted_context
    
    def generate_answer(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful and knowledgeable assistant. Your task is to answer questions based ONLY on the provided context.

Important rules:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Do NOT make up or hallucinate information
4. Always cite your sources by referencing the article titles provided
5. Be accurate and precise in your responses
6. If the question is not related to the provided context, explain this clearly

Provided context:
{context}"""
        
        user_message = f"Question: {query}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def chat(
        self,
        user_message: str,
        top_k: int = 5
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Complete chat pipeline: rewrite query, retrieve context, generate answer.
        
        Args:
            user_message: User input message
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (answer, retrieved_chunks, sources)
        """
        # Step 1: Rewrite query
        rewritten_query = self.rewrite_query(user_message)
        
        # Step 2: Retrieve context
        retrieved_chunks, formatted_context = self.retrieve_context(rewritten_query, top_k=top_k)
        
        # Step 3: Generate answer
        answer = self.generate_answer(rewritten_query, formatted_context)
        
        # Step 4: Extract sources
        sources = self.retriever.get_sources(retrieved_chunks)
        
        # Step 5: Update conversation history
        self.conversation_history.append((user_message, answer))
        
        return answer, retrieved_chunks, sources
    
    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """
        Get current conversation history.
        
        Returns:
            List of (user_message, assistant_response) tuples
        """
        return self.conversation_history.copy()
    
    def clear_history(self):
        """
        Clear conversation history.
        """
        self.conversation_history.clear()


if __name__ == "__main__":
    chat = RAGChat()
    
    # Test chat
    print("Testing RAG Chat...\n")
    
    questions = [
        "What is Machine Learning?",
        "When was it developed?"
    ]
    
    for question in questions:
        print(f"User: {question}")
        answer, chunks, sources = chat.chat(question)
        
        print(f"\nAssistant: {answer}")
        print(f"\nRetrieved {len(chunks)} chunks from {len(sources)} sources")
        
        for source in sources:
            print(f"  - {source['title']} ({source['max_similarity']:.1%})")
        print("\n" + "="*80 + "\n")