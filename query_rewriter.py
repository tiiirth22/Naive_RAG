from typing import List, Tuple
from langchain.chat_models import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from config import GROQ_API_KEY, MODEL_NAME


class QueryRewriter:
    """Rewrites conversational queries into standalone questions."""
    
    def __init__(self, model_name: str = MODEL_NAME, api_key: str = GROQ_API_KEY):
        """
        Initialize query rewriter.
        
        Args:
            model_name: Name of the Groq model
            api_key: Groq API key
        """
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.3
        )
    
    def rewrite_query(
        self,
        current_query: str,
        conversation_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Rewrite a query into a standalone question using conversation context.
        
        Args:
            current_query: Current user question
            conversation_history: List of (user_message, assistant_response) tuples
            
        Returns:
            Rewritten standalone question
        """
        if not conversation_history:
            # If no history, return the query as-is
            return current_query
        
        # Build conversation context
        context_lines = []
        for user_msg, assistant_msg in conversation_history[-5:]:  # Last 5 exchanges
            context_lines.append(f"User: {user_msg}")
            context_lines.append(f"Assistant: {assistant_msg}")
        
        conversation_context = "\n".join(context_lines)
        
        # Create prompt for query rewriting
        system_prompt = """You are a helpful assistant that rewrites follow-up questions into standalone questions.

Given a conversation history and a follow-up question, rewrite the question to be standalone.
The rewritten question should include all necessary context from the conversation.

Rules:
1. Preserve the original intent of the question
2. Include relevant context from the conversation history
3. Make the question clear and specific
4. Return ONLY the rewritten question, nothing else
"""
        
        user_prompt = f"""Conversation history:
{conversation_context}

Follow-up question: {current_query}

Rewritten standalone question:"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            rewritten_query = response.content.strip()
            
            return rewritten_query
        except Exception as e:
            print(f"Error rewriting query: {str(e)}")
            # Return original query if rewriting fails
            return current_query


if __name__ == "__main__":
    rewriter = QueryRewriter()
    
    # Test query rewriting
    conversation_history = [
        ("What is LangChain?", "LangChain is a framework for developing applications powered by language models."),
        ("What are its main features?", "LangChain provides tools for prompt management, chains, agents, and memory management.")
    ]
    
    test_query = "When was it created?"
    print(f"Original query: {test_query}")
    
    rewritten = rewriter.rewrite_query(test_query, conversation_history)
    print(f"Rewritten query: {rewritten}")