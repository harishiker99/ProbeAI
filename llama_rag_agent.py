import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
# Add these lines to load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Utility Functions
def get_history_file_path() -> Path:
    """Get the path to the history JSON file."""
    # Create a conversations directory in the current working directory
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)
    
    # Create a timestamped filename for today
    today = datetime.now().strftime("%Y-%m-%d")
    return conversations_dir / f"conversations_{today}.json"

def load_conversation_history() -> List[Dict[str, Any]]:
    """Load the conversation history from JSON file."""
    history_file = get_history_file_path()
    
    if not history_file.exists():
        return []
        
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")
        return []

def get_context_metadata(docs: List = None) -> Dict[str, Any]:
    """
    Extract metadata about the context documents.
    
    Args:
        docs: List of documents used as context
        
    Returns:
        Dictionary containing metadata about the context
    """
    if not docs:
        return {}
        
    try:
        sources = []
        for doc in docs:
            source = {
                "content": doc.page_content,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "file_name": doc.metadata.get("file_name", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "total_chunks": doc.metadata.get("total_chunks")
            }
            sources.append(source)
            
        return {
            "num_sources": len(sources),
            "sources": sources,
            "retrieval_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error extracting context metadata: {str(e)}")
        return {}

def save_conversation_entry(question: str, answer: str, context: str = None, metadata: Dict[str, Any] = None) -> None:
    """
    Save a new conversation entry to the history JSON file.
    
    Args:
        question: The user's question
        answer: The assistant's answer
        context: The context used to generate the answer
        metadata: Additional metadata about the conversation
    """
    history_file = get_history_file_path()
    
    # Load existing history
    try:
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
            
        # Create new entry with simplified structure
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "context_strings": []
        }
        
        # Extract context strings from documents if available
        if metadata and metadata.get('sources'):
            entry["context_strings"] = [
                src['content'] for src in metadata['sources']
                if src.get('content')
            ]
        elif context:
            # If we only have raw context, add it as a single string
            entry["context_strings"] = [context]
        
        # Add to history
        history.append(entry)
        
        # Save updated history with nice formatting
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
        st.success(f"✅ Conversation saved to {history_file}")
        
    except Exception as e:
        st.error(f"Error saving conversation: {str(e)}")

class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name="mxbai-embed-large:335m"):
        """
        Initialize the OllamaEmbedder with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
        """
        self.model_name = model_name
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)  # mxbai-embed-large uses 1024 dimensions
        self.retry_count = 3
        self.dimensions = 1024  # Update dimensions for mxbai model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Ollama."""
        # Handle empty list
        if not texts:
            return []
            
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                # Try multiple times with exponential backoff
                for attempt in range(self.retry_count):
                    try:
                        embedding = self.embed_query(text)
                        batch_embeddings.append(embedding)
                        break
                    except Exception as e:
                        if attempt == self.retry_count - 1:
                            st.error(f"Failed to embed text after {self.retry_count} attempts: {str(e)}")
                            # Return a zero vector as fallback
                            batch_embeddings.append([0.0] * self.dimensions)
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using Ollama."""
        if not text or not text.strip():
            return [0.0] * self.dimensions
            
        # Try multiple times with exponential backoff
        last_error = None
        for attempt in range(self.retry_count):
            try:
                embedding = self.embedder.get_embedding(text)
                # Verify the dimensions
                if len(embedding) != self.dimensions:
                    raise ValueError(f"Expected embedding of dimension {self.dimensions}, got {len(embedding)}")
                return embedding
            except Exception as e:
                last_error = e
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts failed, log error and return zero vector
        st.error(f"Failed to embed query after {self.retry_count} attempts: {str(last_error)}")
        return [0.0] * self.dimensions

# Constants
COLLECTION_NAME = "1024-mxbai-embed-large"


# Streamlit App Initialization
st.title("🦙 Llama RAG Reasoning Agent")

# Session State Initialization
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
if 'qdrant_url' not in st.session_state:
    st.session_state.qdrant_url = os.getenv("QDRANT_URL", "")
if 'model_version' not in st.session_state:
    st.session_state.model_version = "llama3.1:8b"  # Default to llama3.1:8b
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = os.getenv("EXA_API_KEY", "")
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.5  # Lower default threshold for better recall
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True  # RAG is enabled by default


# Sidebar Configuration

# Model Selection
st.sidebar.header("☑️ Model Selection")
model_help = """
Currently configured to use Llama 3.1:8b for both embeddings and RAG.
Make sure you've pulled the model with 'ollama pull llama3.1:8b'
"""
st.sidebar.info(model_help)
st.sidebar.info("ollama serve")

# RAG Mode Toggle
st.sidebar.header("RAG Configuration")
st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

# Clear Chat Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Show API Configuration only if RAG is enabled
if st.session_state.rag_enabled:
    st.sidebar.header("🔑 API Configuration")
    qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key)
    qdrant_url = st.sidebar.text_input("Qdrant URL", 
                                     placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                     value=st.session_state.qdrant_url)

    # Update session state
    st.session_state.qdrant_api_key = qdrant_api_key
    st.session_state.qdrant_url = qdrant_url
    
    # Search Configuration (only shown in RAG mode)
    st.sidebar.header("Search Configuration")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Document Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Lower values will return more documents but might be less relevant. Higher values are more strict."
    )
    
    # Advanced RAG options in expander
    with st.sidebar.expander("Advanced RAG Settings"):
        # Chunk size settings
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of document chunks. Smaller chunks help with more precise retrieval but can lose context."
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=50,
            step=50,
            help="Overlap between chunks. Higher overlap helps maintain context between chunks."
        )
        
        # Advanced retrieval options
        st.subheader("Retrieval Options")
        
        use_mmr = st.checkbox(
            "Use MMR Retrieval", 
            value=True,
            help="Maximum Marginal Relevance balances relevance with diversity of results"
        )
        
        use_reranking = st.checkbox(
            "Use Semantic Reranking", 
            value=True,
            help="Rerank retrieved documents by semantic similarity to query"
        )
        
        use_query_expansion = st.checkbox(
            "Use Query Expansion", 
            value=True,
            help="Enhance queries with additional keywords for better matching"
        )
        
        use_decomposition = st.checkbox(
            "Use Query Decomposition", 
            value=True,
            help="Break complex queries into simpler sub-queries when no results are found"
        )
        
        # Store values in session state
        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = chunk_size
        else:
            st.session_state.chunk_size = chunk_size
            
        if 'chunk_overlap' not in st.session_state:
            st.session_state.chunk_overlap = chunk_overlap
        else:
            st.session_state.chunk_overlap = chunk_overlap
            
        # Store advanced retrieval options
        st.session_state.use_mmr = use_mmr
        st.session_state.use_reranking = use_reranking
        st.session_state.use_query_expansion = use_query_expansion
        st.session_state.use_decomposition = use_decomposition

# Add in the sidebar configuration section, after the existing API inputs

st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", 
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found"
    )
    st.session_state.exa_api_key = exa_api_key
    
    # Optional domain filtering
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", 
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Add conversation history viewer
st.sidebar.header("📚 Conversation History")

# Add a clear history button
col1, col2 = st.sidebar.columns(2)
view_history = col1.button("View History")
if col2.button("Clear History"):
    try:
        history_file = get_history_file_path()
        if history_file.exists():
            os.remove(history_file)
        st.sidebar.success("History cleared successfully!")
    except Exception as e:
        st.sidebar.error(f"Error clearing history: {str(e)}")

if view_history:
    try:
        history = load_conversation_history()
        if history:
            st.sidebar.markdown("### Previous Conversations")
            for i, entry in enumerate(reversed(history), 1):
                q_num = len(history) - i + 1
                
                # Create a unique key for each entry
                st.sidebar.markdown(f"**Q{q_num}:** _{entry['question']}_")
                st.sidebar.markdown(f"**A:** {entry['answer']}")
                st.sidebar.markdown(f"**Time:** {entry['timestamp']}")
                
                # Add a view details button for each entry
                if st.sidebar.button(f"View Full Details #{q_num}", key=f"details_{q_num}"):
                    with st.expander(f"Conversation #{q_num} Details"):
                        st.markdown(f"**Question:** {entry['question']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
                        st.markdown(f"**Time:** {entry['timestamp']}")
                        
                        if entry.get('context_strings'):
                            st.markdown("**Context Used:**")
                            for idx, ctx in enumerate(entry['context_strings'], 1):
                                with st.expander(f"Context String #{idx}"):
                                    st.text(ctx)
                
                st.sidebar.markdown("---")
        else:
            st.sidebar.info("No conversation history found.")
            st.sidebar.markdown(f"History will be saved to: `{get_history_file_path()}`")
    except Exception as e:
        st.sidebar.error(f"Error loading history: {str(e)}")
        if st.sidebar.button("Reset History"):
            try:
                history_file = get_history_file_path()
                if history_file.exists():
                    os.remove(history_file)
                st.sidebar.success("History reset successfully!")
            except Exception as e2:
                st.sidebar.error(f"Error resetting history: {str(e2)}")

# Search Configuration moved inside RAG mode check


# Utility Functions
def init_qdrant() -> QdrantClient | None:
    """Initialize Qdrant client with configured settings.

    Returns:
        QdrantClient: The initialized Qdrant client if successful.
        None: If the initialization fails.
    """
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


# Document Processing Functions
def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Use chunk settings from session state if available
            chunk_size = st.session_state.get('chunk_size', 1000)
            chunk_overlap = st.session_state.get('chunk_overlap', 50)
            
            # Use smaller chunks with more overlap for better semantic retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add chunk index metadata to help with context
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            st.success(f"PDF processed into {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}")
            return chunks
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


# Vector Store Management
def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  # mxbai-embed-large:335m embedding dimension
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=OllamaEmbeddings()
        )
        
        # Add documents
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None

def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.1:8b"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="Llama RAG Agent",
        model=Ollama(id="llama3.1:8b"),
        # model=Ollama(id="deepseek-r1:1.5b"),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.

        When asked a question:
        - Analyze the question and answer the question with what you know.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        
        Always maintain high accuracy and clarity in your responses and ensure you are using the correct context and make it concise in plain text.
        """,
        show_tool_calls=True,
        markdown=True,
    )

def check_document_relevance(query: str, vector_store, threshold: float = 0.5, use_mmr: bool = True) -> tuple[bool, List]:
    """
    Check document relevance using hybrid search (semantic + lexical) with MMR.
    
    Args:
        query: The user's query
        vector_store: The vector store containing documents
        threshold: Minimum similarity score threshold
        use_mmr: Whether to use MMR retrieval
        
    Returns:
        tuple: (found_relevant_docs, relevant_docs)
    """
    if not vector_store:
        return False, []
    
    if not query or not query.strip():
        return False, []
    
    # Get enhanced query by appending keywords
    enhanced_query = query
    
    # Create a hybrid retriever with both semantic and keyword search
    try:
        # 1. First try MMR retrieval for diverse results
        if use_mmr:
            try:
                mmr_retriever = vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={
                        "k": 8,  # Fetch more candidates
                        "fetch_k": 20,  # Consider a larger pool
                        "lambda_mult": 0.7,  # 0.7 favors relevance, 0.3 favors diversity
                    }
                )
                mmr_docs = mmr_retriever.get_relevant_documents(enhanced_query)
                
                if len(mmr_docs) >= 3:
                    st.info(f"Found {len(mmr_docs)} relevant documents using MMR retrieval")
                    return bool(mmr_docs), mmr_docs
            except Exception as e:
                st.warning(f"MMR retrieval failed: {str(e)}")
        
        # 2. Semantic search with lower threshold
        semantic_retriever = vector_store.as_retriever(
            search_type="similarity",  # Use similarity instead of threshold
            search_kwargs={"k": 10}  # Return more documents
        )
        
        # Perform semantic search
        semantic_docs = semantic_retriever.get_relevant_documents(enhanced_query)
        
        # If we have enough semantic results, we can return them
        if semantic_docs:
            st.info(f"Found {len(semantic_docs)} relevant documents using semantic search")
            return bool(semantic_docs), semantic_docs
        
        # 3. Try hybrid search (semantic + BM25)
        # Get all documents for BM25
        all_docs = get_all_documents(vector_store)
        
        if not all_docs:
            # If we couldn't get all documents but have semantic results
            if semantic_docs:
                st.info(f"Found {len(semantic_docs)} documents using semantic search only")
                return bool(semantic_docs), semantic_docs
            return False, []
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 10
        
        # Perform BM25 search
        keyword_docs = bm25_retriever.get_relevant_documents(enhanced_query)
        
        # 4. Try falling back to lower threshold for similarity
        if not semantic_docs:
            try:
                fallback_retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 15, "score_threshold": max(0.3, threshold - 0.2)}  # Much lower threshold
                )
                semantic_docs = fallback_retriever.get_relevant_documents(enhanced_query)
                if semantic_docs:
                    st.info(f"Found {len(semantic_docs)} documents using fallback similarity search with lower threshold")
            except Exception as e:
                st.warning(f"Fallback similarity search failed: {str(e)}")
        
        # Combine documents if we have both types
        if semantic_docs and keyword_docs:
            # Create ensemble retriever combining both approaches
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            # Get combined results
            docs = ensemble_retriever.get_relevant_documents(enhanced_query)
            
            if docs:
                st.info(f"Hybrid search retrieved {len(docs)} relevant documents")
                return bool(docs), docs
        
        # 5. Just return any documents we have in order of preference
        if semantic_docs:
            st.info(f"Using semantic search results ({len(semantic_docs)} documents)")
            return bool(semantic_docs), semantic_docs
        
        if keyword_docs:
            st.info(f"Using keyword search results ({len(keyword_docs)} documents)")
            return bool(keyword_docs), keyword_docs
        
        return False, []
    
    except Exception as e:
        # Fallback to regular semantic search if everything fails
        st.warning(f"All advanced searches failed, trying simple retrieval: {str(e)}")
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity",  # Simple similarity
                search_kwargs={"k": 10}
            )
            docs = retriever.get_relevant_documents(query)
            return bool(docs), docs
        except Exception as e2:
            st.error(f"All retrieval methods failed: {str(e2)}")
            return False, []

def get_all_documents(vector_store):
    """
    Get all documents from the vector store.
    This is needed for the BM25 retriever.
    
    Args:
        vector_store: The vector store
        
    Returns:
        List of documents
    """
    if not vector_store:
        return []
        
    try:
        # Get all document IDs from Qdrant collection
        if not hasattr(vector_store, "_client") or not hasattr(vector_store, "_collection_name"):
            st.warning("Vector store doesn't have required attributes for document retrieval")
            return []
            
        qdrant_client = vector_store._client
        collection_name = vector_store._collection_name
        
        # Fetch all points with payload from Qdrant (limited to 1000)
        response = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        # Extract documents from the points
        docs = []
        if not response or not response[0]:
            st.warning("No documents found in the vector store")
            return []
            
        for point in response[0]:
            if point.payload and 'page_content' in point.payload and 'metadata' in point.payload:
                doc = Document(
                    page_content=point.payload['page_content'],
                    metadata=point.payload['metadata']
                )
                docs.append(doc)
        
        if not docs:
            st.warning("Retrieved documents couldn't be parsed correctly")
            
        return docs
    except Exception as e:
        st.error(f"Error fetching all documents: {str(e)}")
        return []

def expand_query(query: str, agent: Agent = None) -> str:
    """
    Expand the original query to improve retrieval.
    
    Args:
        query: The original user query
        agent: The agent to use for expansion
    
    Returns:
        An expanded query with additional terms
    """
    if not query or not query.strip():
        return query
        
    # If no agent provided, do simple keyword extraction
    if not agent:
        # Simple keyword extraction
        # Remove common stop words
        stop_words = {"a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "about", "like", "through"}
        keywords = [word for word in query.lower().split() if word not in stop_words]
        
        # For shorter queries, we can just return the original
        if len(keywords) <= 3:
            return query
            
        # For longer queries, return original + key terms
        return f"{query} {' '.join(keywords[:5])}"
    
    # Use the agent for more advanced query expansion
    try:
        expansion_prompt = f"""
        I need you to expand this search query to improve document retrieval:
        
        ORIGINAL QUERY: {query}
        
        Please identify 3-5 key terms or phrases from this query and generate 2-3 alternative 
        phrasings or related terms that would help find relevant information.
        
        Format your response as:
        KEY TERMS: [comma-separated key terms]
        EXPANDED QUERY: [rewritten query with key terms and alternatives]
        
        Keep your response very short and focused on the query without explanations.
        """
        
        response = agent.run(expansion_prompt).content
        
        # Extract the expanded query part
        if "EXPANDED QUERY:" in response:
            expanded_part = response.split("EXPANDED QUERY:")[1].strip()
            return f"{query} {expanded_part}"
        
        return query  # Fallback to original
        
    except Exception as e:
        print(f"Query expansion error: {str(e)}")
        return query  # Fallback to original

def rerank_documents(docs: List, query: str, embedding_model) -> List:
    """
    Rerank documents based on semantic similarity to query.
    
    Args:
        docs: List of retrieved documents
        query: User query
        embedding_model: Embedding model for computing similarity
    
    Returns:
        Reranked list of documents
    """
    if not docs or not query:
        return docs
    
    try:
        # Get the query embedding
        query_embedding = embedding_model.embed_query(query)
        
        # Score documents
        scored_docs = []
        for doc in docs:
            try:
                # Get document embedding
                doc_embedding = embedding_model.embed_query(doc.page_content)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, doc_embedding)
                
                # Add to scored list
                scored_docs.append((doc, similarity))
            except Exception as e:
                # If embedding fails, add with low score
                print(f"Error embedding document: {str(e)}")
                scored_docs.append((doc, 0.0))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the docs, in sorted order
        return [doc for doc, score in scored_docs]
    
    except Exception as e:
        print(f"Error reranking documents: {str(e)}")
        return docs  # Fall back to original order

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score
    """
    import numpy as np
    
    # Convert to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

# Check if RAG is enabled 
if st.session_state.rag_enabled:
    qdrant_client = init_qdrant()
    
    # File/URL Upload Section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    
    # Process documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    # Display sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

if prompt:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.rag_enabled:

            # Existing RAG flow remains unchanged
            with st.spinner("Evaluating the Query..."):
                try:
                    # Create embeddings for the query
                    embedding_model = OllamaEmbeddings()
                    
                    # Generate query embedding
                    try:
                        query_embedding = embedding_model.embed_query(prompt)
                        st.session_state.query_embedding = query_embedding
                        rewritten_query = prompt
                        
                        with st.expander("Evaluating the query"):
                            st.write(f"User's Prompt: {prompt}")
                            st.write(f"Embedding dimensions: {len(query_embedding)}")
                    except Exception as e:
                        st.error(f"❌ Error generating query embedding: {str(e)}")
                        rewritten_query = prompt
                        
                except Exception as e:
                    st.error(f"❌ Error rewriting query: {str(e)}")
                    rewritten_query = prompt

            # Step 2: Choose search strategy based on force_web_search toggle
            context = ""
            docs = []
            
            if not st.session_state.force_web_search and st.session_state.vector_store:
                # Expand the query to improve retrieval if enabled
                if st.session_state.get('use_query_expansion', True):
                    expanded_query = expand_query(rewritten_query)
                    with st.expander("Query Processing"):
                        st.write(f"**Original Query**: {prompt}")
                        st.write(f"**Enhanced Query**: {expanded_query}")
                else:
                    expanded_query = rewritten_query
                
                # Use the improved document relevance checker with expanded query
                with st.spinner("🔍 Performing advanced document search..."):
                    found_docs, retrieved_docs = check_document_relevance(
                        expanded_query, 
                        st.session_state.vector_store,
                        st.session_state.similarity_threshold,
                        use_mmr=st.session_state.get('use_mmr', True)
                    )
                    
                    if found_docs and retrieved_docs:
                        # Apply re-ranking if enabled and we have enough documents
                        if st.session_state.get('use_reranking', True) and len(retrieved_docs) > 3:
                            with st.spinner("⚖️ Re-ranking documents by relevance..."):
                                docs = rerank_documents(retrieved_docs, prompt, embedding_model)
                                st.success(f"Re-ranked {len(docs)} documents by relevance to your query")
                        else:
                            docs = retrieved_docs
                        
                        # Create context from the top documents
                        context = "\n\n".join([d.page_content for d in docs[:8]])  # Limit to top 8 docs
                        st.success(f"Found {len(docs)} relevant documents matching your query")
                    elif st.session_state.use_web_search:
                        st.info("🔄 No relevant documents found in database, falling back to web search...")
                    elif st.session_state.get('use_decomposition', True):
                        # Try query decomposition as a last resort
                        st.warning("Initial search found no documents. Trying query decomposition...")
                        try:
                            # Break the query into simpler sub-queries
                            keywords = [word for word in prompt.lower().split() if len(word) > 3]
                            if keywords:
                                # Create 2-3 word combinations for simpler queries
                                sub_queries = []
                                if len(keywords) >= 3:
                                    for i in range(len(keywords) - 1):
                                        sub_queries.append(f"{keywords[i]} {keywords[i+1]}")
                                else:
                                    sub_queries = keywords
                                
                                # Try each sub-query
                                all_docs = []
                                for sub_query in sub_queries[:3]:  # Limit to top 3 sub-queries
                                    sub_found, sub_docs = check_document_relevance(
                                        sub_query,
                                        st.session_state.vector_store,
                                        max(0.3, st.session_state.similarity_threshold - 0.2)  # Lower threshold
                                    )
                                    if sub_found and sub_docs:
                                        all_docs.extend(sub_docs)
                                
                                if all_docs:
                                    # Deduplicate
                                    seen_content = set()
                                    unique_docs = []
                                    for doc in all_docs:
                                        if doc.page_content not in seen_content:
                                            unique_docs.append(doc)
                                            seen_content.add(doc.page_content)
                                    
                                    # Re-rank if enabled
                                    if st.session_state.get('use_reranking', True):
                                        docs = rerank_documents(unique_docs, prompt, embedding_model)
                                    else:
                                        docs = unique_docs
                                        
                                    context = "\n\n".join([d.page_content for d in docs[:8]])
                                    st.success(f"Query decomposition found {len(docs)} potentially relevant documents")
                                else:
                                    st.warning("No relevant documents found. Try lowering the similarity threshold or adding more documents.")
                            else:
                                st.warning("No relevant documents found. Try lowering the similarity threshold or adding more documents.")
                        except Exception as e:
                            st.error(f"Error during query decomposition: {str(e)}")
                            st.warning("No relevant documents found. Try lowering the similarity threshold or adding more documents.")
                    else:
                        st.warning("No relevant documents found. Try lowering the similarity threshold or adding more documents.")
            
            # If no docs found or web search forced, try web search if enabled
            if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
                with st.spinner("🔍 Searching the web..."):
                    try:
                        web_search_agent = get_web_search_agent()
                        web_results = web_search_agent.run(rewritten_query).content
                        if web_results:
                            context = f"Web Search Results:\n{web_results}"
                            if st.session_state.force_web_search:
                                st.info("ℹ️ Using web search as requested via toggle.")
                            else:
                                st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                    except Exception as e:
                        st.error(f"❌ Web search error: {str(e)}")

            # Step 4: Generate response using the RAG agent
            with st.spinner("🤖 Thinking..."):
                try:
                    rag_agent = get_rag_agent()
                    
                    if context:
                        full_prompt = f"""
Please answer the question accurately based ONLY on the provided context information.

QUESTION: {prompt}

CONTEXT:
{context}

INSTRUCTIONS:
1. Focus on finding SPECIFIC and PRECISE information from the context
2. Look for exact numbers, percentages, dates, and statistics that directly answer the question
3. If there are multiple sources or conflicting information, mention all of them
4. If the exact answer is not in the context, clearly state that the specific information is not available
5. DO NOT make up or infer information not present in the context
6. Cite specific sections of the context when possible

Answer the question as thoroughly and specifically as possible given ONLY the information in the context.
"""
                    else:
                        full_prompt = f"Original Question: {prompt}\n"
                        st.info("ℹ️ No relevant information found in documents or web search.")

                    response = rag_agent.run(full_prompt)
                    response_content = response.content
                    
                    # Save conversation to history
                    metadata = {
                        "rag_enabled": st.session_state.rag_enabled,
                        "similarity_threshold": st.session_state.similarity_threshold,
                        "use_web_search": st.session_state.use_web_search,
                        "force_web_search": st.session_state.force_web_search
                    }
                    
                    if docs:
                        metadata.update(get_context_metadata(docs))
                        
                    save_conversation_entry(
                        question=prompt,
                        answer=response_content,
                        context=context,
                        metadata=metadata
                    )
                    
                    # Add assistant response to history
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(response_content)
                        
                        # Show sources if available
                        if not st.session_state.force_web_search and 'docs' in locals() and docs:
                            with st.expander("🔍 See document sources"):
                                # Sort documents by relevance or chunk order if they're from the same PDF
                                sorted_docs = sorted(docs, 
                                    key=lambda x: (
                                        x.metadata.get('file_name', ''),  # Group by filename
                                        x.metadata.get('chunk_index', 0)  # Then by chunk index
                                    )
                                )
                                
                                # Group documents by source
                                sources = {}
                                for doc in sorted_docs:
                                    source_type = doc.metadata.get("source_type", "unknown")
                                    source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                    if source_name not in sources:
                                        sources[source_name] = []
                                    sources[source_name].append(doc)
                                
                                # Display documents grouped by source
                                for source_name, source_docs in sources.items():
                                    st.subheader(f"Source: {source_name}")
                                    for i, doc in enumerate(source_docs, 1):
                                        source_type = doc.metadata.get("source_type", "unknown")
                                        source_icon = "📄" if source_type == "pdf" else "🌐"
                                        chunk_index = doc.metadata.get('chunk_index')
                                        total_chunks = doc.metadata.get('total_chunks')
                                        chunk_info = f" (Chunk {chunk_index+1}/{total_chunks})" if chunk_index is not None and total_chunks is not None else ""
                                        
                                        st.markdown(f"**{source_icon} Passage {i}{chunk_info}:**")
                                        st.markdown(f"```\n{doc.page_content}\n```")

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

    else:
        # Simple mode without RAG
        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                web_search_agent = get_web_search_agent() if st.session_state.use_web_search else None
                
                # Handle web search if forced or enabled
                context = ""
                if st.session_state.force_web_search and web_search_agent:
                    with st.spinner("🔍 Searching the web..."):
                        try:
                            web_results = web_search_agent.run(prompt).content
                            if web_results:
                                context = f"Web Search Results:\n{web_results}"
                                st.info("ℹ️ Using web search as requested.")
                        except Exception as e:
                            st.error(f"❌ Web search error: {str(e)}")
                
                # Generate response
                if context:
                    full_prompt = f"""
Please answer the question accurately based ONLY on the provided context information.

QUESTION: {prompt}

CONTEXT:
{context}

INSTRUCTIONS:
1. Focus on finding SPECIFIC and PRECISE information from the context
2. Look for exact numbers, percentages, dates, and statistics that directly answer the question
3. If there are multiple sources or conflicting information, mention all of them
4. If the exact answer is not in the context, clearly state that the specific information is not available
5. DO NOT make up or infer information not present in the context
6. Cite specific sections of the context when possible

Answer the question as thoroughly and specifically as possible given ONLY the information in the context.
"""
                else:
                    full_prompt = f"Original Question: {prompt}\n"
                    st.info("ℹ️ No relevant information found in documents or web search.")

                response = rag_agent.run(full_prompt)
                response_content = response.content
                
                # Extract thinking process and final response
                import re
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, response_content, re.DOTALL)
                
                if think_match:
                    thinking_process = think_match.group(1).strip()
                    final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
                else:
                    thinking_process = None
                    final_response = response_content
                
                # Add assistant response to history (only the final response)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    if thinking_process:
                        with st.expander("See thinking process"):
                            st.markdown(thinking_process)
                    st.markdown(final_response)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                
else:
    st.warning("Llama 3.1:8b reasoning locally, toggle the RAG mode to upload documents!")