# SCA Assist

End-to-end implementation of a RAG (Retrieval-Augmented Generation) based document search assistant with accurate retrieval and AI-powered question answering.

## Features

- **Document Ingestion**: Upload and process PDF documents
- **Chunking**: Recursive text splitting for optimal context windows
- **Vector Embeddings**: OpenAI's `text-embedding-3-large` model (3072 dimensions)
- **Vector Store**: MongoDB for storing document embeddings
- **Similarity Search**: Cosine similarity for document retrieval
- **Smart Query Router**: Classifies queries to skip RAG for greetings/off-topic
- **Query Reranking**: OpenAI-powered reranker for precision boost
- **RAG Question Answering**: Context-aware answers using OpenAI GPT models
- **Prompt Guardrails**: Out-of-context detection and response rules
- **Structured Output**: Pydantic-based structured LLM responses
- **Prompt Configuration**: YAML-based prompt management
- **REST API**: FastAPI endpoints for all operations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Web Framework | FastAPI |
| LLM | OpenAI GPT-4o-mini |
| Query Router | OpenAI GPT-4o-mini (Structured Output) |
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
│   │   ├── rag.yaml            # RAG Q&A prompts with guardrails
│   │   └── router.yaml         # Query router prompts & responses
│   ├── services/
│   │   ├── ingest_service.py   # Document loading and chunking
│   │   ├── openai_service.py   # OpenAI chat, embeddings & structured output
│   │   ├── mongodb_service.py  # MongoDB vector store operations
│   │   ├── rag_service.py      # RAG business logic
│   │   ├── reranker_service.py # Query reranking for precision boost
│   │   └── query_router_service.py # Smart query classification
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

## Docker Usage

To run the application and MongoDB using Docker Compose:

### Build and Start Containers
```bash
docker-compose up --build
```

### Stop Containers
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Restart Containers
```bash
docker-compose restart
```

### Run in Detached Mode (background)
```bash
docker-compose up -d
```

### Rebuild Only the API Service
```bash
docker-compose up --build api
```

These commands will start both the FastAPI server and MongoDB in isolated containers, using the environment variables from your `.env` file. Make sure your `.env` uses `MONGODB_URI=mongodb://mongodb:27017` for Docker Compose.

NOTE : If you're using docker then no need of installation step just add env var in .env only

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

  Create a `.env` file in the project root. Use the correct MongoDB URI for your environment:

  **For local development:**
  ```env
  DEBUG=true
  OPENAI_API_KEY=your-openai-api-key-here
  MONGODB_URI=mongodb://localhost:27017
  ```

  **For Docker Compose:**
  ```env
  DEBUG=true
  OPENAI_API_KEY=your-openai-api-key-here
  MONGODB_URI=mongodb://mongodb:27017
  ```

5. **Start MongoDB**

  Make sure MongoDB is running on `localhost:27017` for local development, or use Docker Compose to start both API and MongoDB containers.

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
# Document query (goes through full RAG pipeline)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "k": 5}'

# Greeting (handled directly by router - no RAG)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi"}'
# Response: "Hello! I'm your document assistant..."

# Without reranking
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "use_reranker": false}'
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

| Variable        | Description                        | Required |
|-----------------|------------------------------------|----------|
| `DEBUG`         | Enable debug mode (true/false)     | Yes      |
| `OPENAI_API_KEY`| OpenAI API key                     | Yes      |
| `MONGODB_URI`   | MongoDB connection URI             | Yes      |

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
| MongoDB Host | localhost (local) / mongodb (docker) |
| MongoDB Port | 27017 |
| Database Name | sca_assist |
| Vector Collection | documents |

## Smart Query Router

The query router classifies incoming queries to optimize processing:

### Query Classifications

| Type | Example | Action |
|------|---------|--------|
| `greeting` | "Hi", "Hello", "Thanks" | Direct friendly response (no RAG) |
| `document_query` | "What is attention?" | Full RAG pipeline |
| `off_topic` | "What's the weather?" | Polite rejection (no RAG) |
| `clarification` | "What can you do?" | Help/usage info (no RAG) |

### How It Works

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │ ─── Greeting ────► Direct Response
│  (Classifier)   │ ─── Off-topic ───► Rejection
│                 │ ─── Clarification ► Help Info
└────────┬────────┘
         │
    Document Query
         │
         ▼
┌─────────────────┐
│  Vector Search  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reranker     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Answer    │
└─────────────────┘
```

### Benefits

- **Cost Savings**: Skip expensive embeddings/LLM calls for simple queries
- **Faster Response**: Instant responses for greetings
- **Better UX**: Appropriate responses for each query type

## Query Reranking

The reranker improves search precision by re-scoring initial vector search results using GPT-4o-mini:

1. **Initial Search**: Fetches top 20 documents using vector similarity
2. **Reranking**: GPT-4o-mini scores each document's relevance (0-10 scale)
3. **Final Results**: Returns top K documents sorted by rerank score

### Scoring Guidelines

| Score | Meaning |
|-------|--------|
| 9-10 | Directly answers query |
| 7-8 | Highly relevant |
| 5-6 | Somewhat related |
| 3-4 | Minor relevance |
| 1-2 | Very weak relevance |
| 0 | Completely irrelevant |

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
| `router.yaml` | Query router prompts & canned responses |
| `reranker.yaml` | Reranker system/user prompts + model settings |
| `rag.yaml` | RAG Q&A system/user prompts with guardrails |

### Prompt Guardrails

The RAG prompts include strict guardrails:

- **Context-Only Answers**: Use ONLY information from documents
- **Out-of-Context Detection**: Reject unrelated questions
- **No Hallucination**: Never make up information
- **Citation Required**: Always cite page numbers
- **Off-Topic Rejection**: Politely decline irrelevant queries

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
