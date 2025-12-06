# SCA Assist

End-to-end implementation of a RAG (Retrieval-Augmented Generation) based document search assistant with accurate retrieval and AI-powered question answering.

## Features

- **Document Ingestion**: Upload and process PDF documents
- **Chunking**: Recursive text splitting for optimal context windows
- **Vector Embeddings**: OpenAI's `text-embedding-3-large` model (3072 dimensions)
- **Vector Store**: MongoDB for storing document embeddings
- **Similarity Search**: Cosine similarity for document retrieval
- **Query Reranking**: OpenAI-powered reranker for precision boost
- **RAG Question Answering**: Context-aware answers using OpenAI GPT models
- **Structured Output**: Pydantic-based structured LLM responses
- **Prompt Configuration**: YAML-based prompt management
- **REST API**: FastAPI endpoints for all operations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Web Framework | FastAPI |
| LLM | OpenAI GPT-4o-mini |
| Reranker | OpenAI GPT-4o-mini (Structured Output) |
| Embeddings | OpenAI text-embedding-3-large |
| Vector Store | MongoDB |
| Document Processing | LangChain |
| PDF Parsing | PyPDF |
| Prompt Config | YAML |

## Project Structure

```
sca-assist/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── routes.py           # API endpoint definitions
│   ├── config/
│   │   └── settings.py         # Pydantic settings configuration
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── prompt_config/          # YAML prompt configuration files
│   │   ├── reranker.yaml       # Reranker prompts & settings
│   │   └── rag.yaml            # RAG Q&A prompts
│   ├── services/
│   │   ├── ingest_service.py   # Document loading and chunking
│   │   ├── openai_service.py   # OpenAI chat, embeddings & structured output
│   │   ├── mongodb_service.py  # MongoDB vector store operations
│   │   ├── rag_service.py      # RAG business logic
│   │   └── reranker_service.py # Query reranking for precision boost
│   └── utils/
│       ├── logging.py          # Logging configuration
│       └── prompt_loader.py    # YAML prompt loader utility
├── experiments/
│   └── notebooks/              # Jupyter notebooks for testing
├── logs/                       # Application logs
├── .env                        # Environment variables
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.12+
- MongoDB (local or Atlas)
- OpenAI API Key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jaxayprajapati/sca-assist.git
   cd sca-assist
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn python-multipart
   pip install langchain langchain-openai langchain-community langchain-text-splitters
   pip install pymongo pypdf pydantic-settings pyyaml
   ```

4. **Create environment file**

   Create a `.env` file in the project root:
   ```env
   DEBUG=true
   OPENAI_API_KEY=your-openai-api-key-here
   ```

5. **Start MongoDB**

   Make sure MongoDB is running on `localhost:27017`

## Running the Application

### Start the API Server

```bash
# From project root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or using Python:
```bash
python -m app.main
```

### Access API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root - check if API is running |
| GET | `/health` | Health check for all services |

### Document Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/pdf` | Upload and ingest a PDF file |

### Question Answering

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a question (RAG-based answer) |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Similarity search on documents |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents/count` | Get total document count |
| DELETE | `/documents` | Delete all documents |

## Usage Examples

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/ingest/pdf" \
  -F "file=@document.pdf"
```

### Ask a Question

```bash
# With reranking (default - enabled)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "k": 5, "use_reranker": true}'

# Without reranking
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "k": 5, "use_reranker": false}'
```

### Similarity Search

```bash
# With reranking (default - enabled)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer architecture", "k": 5, "use_reranker": true}'

# Without reranking
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer architecture", "k": 5, "use_reranker": false}'
```

### Get Document Count

```bash
curl "http://localhost:8000/documents/count"
```

### Delete All Documents

```bash
curl -X DELETE "http://localhost:8000/documents"
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEBUG` | Enable debug mode (true/false) | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |

### Default Settings

| Setting | Value |
|---------|-------|
| Chunk Size | 1000 characters |
| Chunk Overlap | 100 characters |
| Embedding Model | text-embedding-3-large |
| Embedding Dimensions | 3072 |
| Chat Model | gpt-4o-mini |
| Reranker Model | gpt-4o-mini |
| Reranker Temperature | 0 |
| MongoDB Host | localhost |
| MongoDB Port | 27017 |
| Database Name | sca_assist |
| Vector Collection | documents |

## Query Reranking

The reranker improves search precision by re-scoring initial vector search results using GPT-4o-mini:

1. **Initial Search**: Fetches top 20 documents using vector similarity
2. **Reranking**: GPT-4o-mini scores each document's relevance (0-10 scale)
3. **Final Results**: Returns top K documents sorted by rerank score

### How It Works

```
Query → Vector Search (20 docs) → Reranker (GPT-4o-mini) → Top K Results
```

### Configuration

Reranker settings are in `app/prompt_config/reranker.yaml`:

```yaml
reranker:
  system: |  # System prompt
  user: |    # User prompt with {query} and {doc_list} placeholders
  model: "gpt-4o-mini"
  temperature: 0
```

## Prompt Configuration

All LLM prompts are externalized to YAML files in `app/prompt_config/`:

| File | Purpose |
|------|--------|
| `reranker.yaml` | Reranker system/user prompts + model settings |
| `rag.yaml` | RAG Q&A system/user prompts |

### Usage

```python
from app.utils.prompt_loader import prompt_loader

# Get a prompt
system = prompt_loader.get_prompt("reranker", "reranker", "system")

# Format a prompt with variables
user = prompt_loader.format_prompt(
    "rag", "qa", "user",
    context=context,
    question=question
)

# Get config values
model = prompt_loader.get_config("reranker", "reranker", "model", "gpt-4o-mini")
```

### Benefits

- **No code changes**: Modify prompts without touching code
- **Version control**: Track prompt changes in git
- **Easy testing**: Quickly iterate on prompts
- **Centralized**: All prompts in one location

## Development

### Running Jupyter Notebooks

```bash
jupyter notebook experiments/notebooks/
```

### Logging

Logs are stored in the `logs/` directory:
- `logs/app.log` - Application logs

Log format includes timestamp, level, filename, line number, function name, and message.

## License

MIT License

## Author

Jaxay Prajapati
