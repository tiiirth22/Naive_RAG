import streamlit as st
import os
from pathlib import Path
from chat import RAGChat
from build_vector_db import initialize_vector_db
from config import DATA_PATH, CHROMA_DB_PATH


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-badge {
        display: inline-block;
        background-color: #e1f5fe;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False


def check_vector_db_exists() -> bool:
    """Check if vector database exists and has data."""
    if not os.path.exists(CHROMA_DB_PATH):
        return False
    
    # Check if ChromaDB files exist
    return len(os.listdir(CHROMA_DB_PATH)) > 0


def build_knowledge_base(keywords_file):
    """Build knowledge base from CSV file."""
    from scrape_wikipedia import scrape_wikipedia_pages, save_scraped_data
    
    with st.spinner("📥 Scraping Wikipedia articles..."):
        scraped_data = scrape_wikipedia_pages(keywords_file)
        save_scraped_data(scraped_data)
        total_articles = sum(len(articles) for articles in scraped_data.values())
        st.success(f"✓ Scraped {total_articles} articles")
    
    with st.spinner("🔧 Building vector database..."):
        db = initialize_vector_db(reset=True)
        chunk_count = db.get_collection_size()
        st.success(f"✓ Built vector database with {chunk_count} chunks")
    
    st.session_state.db_initialized = True
    st.session_state.chat = RAGChat()


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.divider()
        
        st.subheader("📚 Knowledge Base")
        
        # Display current status
        db_exists = check_vector_db_exists()
        if db_exists:
            st.success("✓ Knowledge base ready")
            if st.session_state.chat:
                doc_count = st.session_state.chat.retriever.collection.count()
                st.metric("Indexed Chunks", doc_count)
        else:
            st.warning("⚠ No knowledge base found")
        
        st.divider()
        
        # File upload section
        st.subheader("📄 Upload Keywords")
        st.caption("CSV format: keyword, pages")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = "temp_keywords.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔨 Build Knowledge Base", use_container_width=True, type="primary"):
                try:
                    build_knowledge_base(temp_path)
                except Exception as e:
                    st.error(f"Error building knowledge base: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chat:
                st.session_state.chat.clear_history()
            st.rerun()
        
        st.divider()
        st.caption("🔧 Powered by LangChain + ChromaDB + Groq")
    
    # Main content
    st.title("💬 RAG Chatbot")
    
    # Check if knowledge base is ready
    if not check_vector_db_exists():
        st.info(
            "📚 **No knowledge base found**\n\n"
            "Please upload a CSV file with keywords in the sidebar to build the knowledge base.",
            icon="ℹ️"
        )
        st.stop()
    
    # Initialize chat if not already done
    if not st.session_state.chat:
        st.session_state.chat = RAGChat()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(
                                f"[{source['title']}]({source['url']})",
                                unsafe_allow_html=False
                            )
                        with col2:
                            st.caption(f"{source['max_similarity']:.0%}")
    
    # Chat input
    if user_input := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, chunks, sources = st.session_state.chat.chat(
                        user_input,
                        top_k=top_k
                    )
                    
                    st.markdown(answer)
                    
                    # Display sources
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(sources, 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(
                                    f"[{source['title']}]({source['url']})",
                                    unsafe_allow_html=False
                                )
                            with col2:
                                st.caption(f"{source['max_similarity']:.0%}")
                    
                    # Store message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()