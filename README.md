# RAG Chatbot Application

A complete Retrieval-Augmented Generation (RAG) chatbot application built with Python, featuring a modern Streamlit interface.

## 🏗 Architecture

```
User Question
    ↓
Conversation History
    ↓
Question Rewriter (LLM-powered)
    ↓
Vector Search (ChromaDB)
    ↓
Retrieve Relevant Chunks
    ↓
LLM Answer Generation
    ↓
Streamlit Chat UI
```

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Language Model**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: Sentence Transformers (BAAI/bge-small-en-v1.5)
- **Vector DB**: ChromaDB
- **LLM Framework**: LangChain
- **Data Source**: Wikipedia
- **Python Version**: 3.8+

## ✨ Features

### 1. Data Collection
- CSV-based keyword input for Wikipedia article selection
- Automatic Wikipedia scraping with metadata preservation
- Error handling for disambiguation and missing pages
- JSON-based data storage

### 2. Intelligent Chunking
- Adaptive text chunking (~1000 characters per chunk)
- Overlap preservation (100 characters) for context continuity
- Metadata preservation (article title, source URL, chunk ID)

### 3. Vector Embeddings
- State-of-the-art sentence embeddings (BAAI/bge-small-en-v1.5)
- Cosine similarity-based retrieval
- Efficient ChromaDB backend

### 4. Conversational AI
- Query rewriting using LLM for better context understanding
- Chat history management
- Follow-up question handling
- Source citation

### 5. Modern UI
- Real-time chat interface
- Source attribution with similarity scores
- Knowledge base statistics
- Customizable retrieval settings
- Clean and responsive design

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [console.groq.com](https://console.groq.com))
- 2GB+ RAM recommended
- 1GB+ disk space for ChromaDB

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tiiirth22/Naive_RAG.git
cd Naive_RAG
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- langchain
- chromadb
- sentence-transformers
- groq
- wikipedia
- python-dotenv
- And other required packages

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_DB_PATH=./chroma_db
DATA_PATH=./data
```

## 📚 Building the Knowledge Base

### Step 1: Create Keywords CSV

Create a `keywords.csv` file:

```csv
keyword,pages
Python,1
Machine Learning,1
LangChain,1
Artificial Intelligence,2
Data Science,1
```

**Format**:
- `keyword`: Topic to search on Wikipedia
- `pages`: Number of pages to scrape (per keyword)

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 3: Build Knowledge Base

1. In the sidebar, click "Upload Keywords"
2. Select your `keywords.csv` file
3. Click "🔨 Build Knowledge Base"
4. Wait for scraping and indexing to complete

**What happens**:
- Wikipedia articles are scraped
- Text is split into chunks (~1000 chars each)
- Embeddings are generated and stored in ChromaDB
- Knowledge base is ready for querying

## 💬 Using the Chatbot

1. Once the knowledge base is built, type your question
2. The chatbot will:
   - Rewrite your question for clarity
   - Retrieve relevant chunks from the knowledge base
   - Generate an answer using the Groq LLM
   - Display sources and similarity scores

### Example Interactions

**Q1**: "What is Machine Learning?"
- LLM retrieves relevant Wikipedia sections
- Generates comprehensive answer
- Shows sources

**Q2**: "When was it developed?"
- System detects this is a follow-up
- Rewrites to: "When was Machine Learning developed?"
- Retrieves context-aware results
- Provides answer with citations

## 📁 Project Structure

```
Naive_RAG/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration and environment variables
├── scrape_wikipedia.py    # Wikipedia scraping module
├── build_vector_db.py     # Vector database creation
├── retrieve.py            # Document retrieval system
├── query_rewriter.py      # Query rewriting with LLM
├── chat.py                # RAG chat orchestration
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── data/                 # Scraped Wikipedia data (JSON)
├── chroma_db/            # ChromaDB vector store
└── keywords.csv          # Your keywords file (create this)
```

## 🔧 Configuration

### Customizing Parameters

Edit `config.py` to customize:

```python
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 100            # Character overlap
TOP_K_CHUNKS = 5               # Chunks to retrieve
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Embedding model
```

### Advanced Settings in UI

- **Number of chunks to retrieve**: 1-10 (default: 5)
- **Chat history**: Clear with button in sidebar
- **Temperature**: Affects response creativity (default: 0.7)

## 🧪 Testing Individual Components

### Test Wikipedia Scraping

```bash
python scrape_wikipedia.py keywords.csv
```

### Test Vector Database

```bash
python build_vector_db.py
```

### Test Document Retrieval

```bash
python retrieve.py
```

### Test Query Rewriting

```bash
python query_rewriter.py
```

### Test Chat System

```bash
python chat.py
```

## 📊 Performance Metrics

- **Embedding Generation**: ~1-2 seconds per chunk
- **Retrieval Latency**: <100ms (local ChromaDB)
- **LLM Response Time**: 2-5 seconds (depends on query complexity)
- **Total Response Time**: 3-7 seconds per query

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not set"

**Solution**: 
1. Create `.env` file with your API key
2. Ensure the key is correct and active
3. Restart the app

### Issue: "No module named 'chromadb'"

**Solution**:
```bash
pip install --upgrade chromadb
```

### Issue: Wikipedia scraping is slow

**Solution**:
- Reduce number of pages per keyword
- Check internet connection
- Wikipedia API rate limiting may apply

### Issue: Vector database out of memory

**Solution**:
- Reduce chunk size in `config.py`
- Scrape fewer articles
- Use a machine with more RAM

### Issue: "LLM Error: Rate limit exceeded"

**Solution**:
- Wait a few minutes before querying again
- Check Groq API quota
- Consider upgrading API tier

## 🌐 Deployment

### GitHub Codespaces

```bash
# In Codespaces terminal:
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Access at: `https://<codespace-name>-8501.app.github.dev`

### Local Development

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### Production Deployment

**Note**: For production, consider:
- Using PostgreSQL instead of local storage
- Implementing authentication
- Adding rate limiting
- Using a cloud-hosted vector database
- Scaling LLM API calls

## 📝 Example Workflow

```
1. Create keywords.csv with topics
2. Run: streamlit run app.py
3. Upload keywords.csv in sidebar
4. Click "Build Knowledge Base"
5. Wait for indexing (~1-5 minutes depending on articles)
6. Start asking questions
7. Receive answers with sources and similarity scores
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **ChromaDB** for vector database
- **Sentence Transformers** for embeddings
- **LangChain** for LLM orchestration
- **Streamlit** for beautiful UI
- **Wikipedia** for knowledge content

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] PDF/Document upload support
- [ ] Real-time collaboration
- [ ] Custom knowledge base creation
- [ ] Answer quality metrics
- [ ] Export conversations
- [ ] API endpoint for integration
- [ ] Advanced analytics dashboard

---

**Built with ❤️ for the open-source community**