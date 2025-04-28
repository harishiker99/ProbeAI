# ProbeAI

**ProbeAI** is a reasoning agent framework designed to utilize RAG (Retrieval-Augmented Generation) techniques. It leverages AI models and document processing to provide intelligent responses by combining document retrieval, embeddings, and reasoning capabilities. 

## Features

- **DeepSeek RAG Agent**: Implements a reasoning agent using the DeepSeek R1 model with document embeddings and similarity-based retrieval.
- **Llama RAG Agent**: An alternative reasoning agent using Llama embeddings and enhanced query expansion and retrieval features.
- **Document Processing**: Supports PDF processing and web scraping for document ingestion.
- **Web Search Fallback**: Includes web search capabilities for scenarios where documents lack relevant information.
- **Conversation History Management**: Saves and retrieves conversation history with metadata for context tracking.
- **Customizable Configuration**: Offers settings for RAG mode, similarity thresholds, query expansion, and more.

## Installation

To set up the environment, clone this repository and install the required dependencies:

```bash
git clone https://github.com/harishiker99/ProbeAI.git
cd ProbeAI
pip install -r requirements.txt
```

## Requirements

The following Python libraries are required and specified in the `requirements.txt`:

- `streamlit`
- `beautifulsoup4`
- `bs4`
- `agno`
- `langchain-community`
- `langchain`
- `langchain-qdrant`
- `qdrant-client`
- `langchain-core`
- `python-dotenv`
- `ollama`
- `exa_py`
- `sentence-transformers`
- `pypdf`

Ensure you have Python 3.8+ installed to run the application.

## Usage

### Running the Application

Start the Streamlit application:

```bash
streamlit run deepseek_rag_agent.py
```

or

```bash
streamlit run llama_rag_agent.py
```

### DeepSeek RAG Agent

The `deepseek_rag_agent.py` script initializes a reasoning agent using the DeepSeek R1 model. It supports:

- Embedding and storing document vectors using the `SentenceTransformerEmbedder`.
- PDF and web document processing for context-aware responses.
- RAG (Retrieval-Augmented Generation) mode for improved accuracy.

### Llama RAG Agent

The `llama_rag_agent.py` script sets up a reasoning agent using Llama embeddings. It includes:

- Query expansion with keywords to improve retrieval accuracy.
- Advanced document re-ranking with Maximum Marginal Relevance (MMR).
- Document retrieval with a hybrid search (semantic + keyword).

### Configuration

Both agents support customization via Streamlit's sidebar:

- **Model Selection**: Choose between different model versions.
- **RAG Mode**: Enable or disable retrieval-augmented generation mode.
- **API Configuration**: Enter API keys for Qdrant and Exa AI services.
- **Similarity Threshold**: Adjust thresholds for document similarity scoring.
- **Web Search Configuration**: Enable fallback to web search and configure domains to search.

### Document Processing

- **PDF Processing**: Upload PDF files to extract content and store document embeddings.
- **Web Processing**: Enter web URLs to fetch and process content, adding metadata for retrieval.

## Architecture

1. **Document Embedding**: Uses `SentenceTransformerEmbedder` for DeepSeek and `OllamaEmbedder` for Llama.
2. **Vector Store**: Built using Qdrant, a high-performance vector database.
3. **Retrieval**: Combines similarity scoring, keyword search, and semantic re-ranking.
4. **Reasoning Agent**: Configured with instructions for context-aware answers, leveraging both document and web search results.

## Environment Variables

Create a `.env` file in the root directory and add the following variables:

```env
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_URL=<your-qdrant-url>
EXA_API_KEY=<your-exa-api-key>
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your contribution.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or feedback, reach out to the repository owner, [harishiker99](https://github.com/harishiker99).
