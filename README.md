# Multi-Modal Legal AI Chatbot System

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [How It Works](#how-it-works)
8. [API Endpoints](#api-endpoints)
9. [Usage Examples](#usage-examples)
10. [Performance & Metrics](#performance--metrics)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

**Multi-Modal Legal AI Chatbot** is a production-ready artificial intelligence system designed to intelligently process, understand, and answer complex legal questions by analyzing legal documents, contracts, and case materials.

### Problem Statement
Legal professionals struggle with:
- Time-consuming document review and analysis
- Finding relevant clauses and precedents across large document sets
- Ensuring consistency in legal interpretation
- Risk of missing critical information due to document complexity

### Solution
Our system provides an intelligent chatbot that:
- **Ingests** multi-modal legal documents (PDFs with text and images)
- **Retrieves** relevant information using hybrid search (semantic + keyword)
- **Validates** retrieved documents for relevance
- **Generates** comprehensive answers with automatic citations
- **Checks** for hallucinations and ensures factual accuracy
- **Provides** metrics and confidence scores for transparency

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Chatbot)                      │
│              (Web-based Chat Interface via FastAPI)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                           │
│  • POST /chat/message    • GET /chat/history                    │
│  • POST /chat/session    • GET /chatbot                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  Retrieval   │ │ Self-Correct │ │   Answer     │
        │  Pipeline    │ │   Loop       │ │  Generation  │
        └──────────────┘ └──────────────┘ └──────────────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Query Expansion │  │  Document Grader │  │ Hallucination    │
│  & Optimization  │  │  (Relevance      │  │ Checker          │
│                  │  │   Assessment)    │  │ (Answer Validate)│
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Hybrid Retriever │  │  Query Rewriter  │  │ Vector Database  │
│ (Semantic + BM25)│  │ (Auto-Retry)     │  │ (Chroma)         │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │
        │  ┌─────────────────────────────┐
        └──▶│  Ingestion Pipeline        │
           │  • PDF Parsing             │
           │  • Image Extraction        │
           │  • Vision Processing       │
           │  • Text Cleaning           │
           │  • Intelligent Chunking    │
           │  • Embedding Generation    │
           └─────────────────────────────┘
```

### 1️⃣ Data Ingestion Layer

**Purpose:** Transform raw legal documents into searchable knowledge base

**Components:**
- **PDF Parser** (`pdf_parser.py`)
  - Supports Marker and Unstructured.io parsers with fallback
  - Extracts text while preserving structure
  - Detects and extracts images for vision processing
  - Handles complex layouts, tables, and multi-column text

- **Image Processor** (`vision_processor.py`)
  - Uses GPT-4V for intelligent image description
  - Async batch processing with concurrency limits
  - Detects and skips low-quality images
  - Generates context-aware captions for legal diagrams

- **Text Cleaner** (`text_cleaner.py`)
  - Boilerplate removal (headers, footers, page numbers)
  - Whitespace normalization
  - Special character handling
  - Legal-specific cleaning (removes disclaimer patterns)

- **Chunking** (`chunking.py`)
  - Section-aware intelligent splitting
  - Detects legal clause boundaries
  - Maintains context across chunks
  - Fallback character-based chunking for non-standard documents

- **Vector Store** (`vector_store.py`)
  - Chroma database for efficient similarity search
  - OpenAI embeddings (text-embedding-3-small)
  - Batch processing and persistence
  - Collection management and statistics

### 2️⃣ Retrieval Pipeline

**Purpose:** Find most relevant documents from knowledge base

**Components:**
- **Query Expansion** (`query_expander.py`)
  - Legal ontology with 200+ terms
  - 8 query types (general, clause, party, obligation, etc.)
  - Synonym expansion and context injection
  - Query optimization for backend search

- **Hybrid Retriever** (`retriever.py`)
  - **Semantic Search:** Vector similarity using embeddings
  - **Keyword Search:** BM25 probabilistic ranking
  - Score fusion combining both methods
  - Advanced filtering by metadata
  - Re-ranking by relevance

- **Retrieval Integration** (`retrieval_integration.py`)
  - High-level API for different search types
  - Answer-specific question retrieval
  - Clause comparison and analysis
  - Category-based search

### 3️⃣ Self-Correction Loop

**Purpose:** Ensure high-quality, accurate answers through validation and retry

**Components:**
- **Document Grader** (`selfCorrection/grader.py`)
  - LLM-based relevance assessment
  - 3-tier grading: RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT
  - Keyword coverage analysis
  - Legal clause detection
  - Quality scoring

- **Query Rewriter** (`rewriter.py`)
  - 5 rewriting strategies: EXPAND, SIMPLIFY, REPHRASE, FOCUS, DECOMPOSE
  - Handles failed retrievals
  - Fallback rewriting if API fails
  - Retry strategy with configurable max attempts

- **Hallucination Checker** (`selfCorrection/hallucination_checker.py`)
  - Validates answers against source documents
  - Detects fabricated information
  - 4-level severity: NONE, MINOR, MODERATE, SEVERE
  - Claim extraction and validation
  - Source grounding verification

- **Answer Generator** (`answer_generator.py`)
  - Context injection with relevant chunks
  - Multi-modal support (text + images)
  - Automatic citation extraction
  - Multiple output formats (markdown, JSON, HTML)

- **Orchestrator** (`self_correction_orchestrator.py`)
  - LangGraph state machine
  - Workflow: Retrieve → Grade → Rewrite → Generate → Check → Complete
  - Automatic retry on failures
  - Metrics tracking for each stage

### 4️⃣ API Layer (FastAPI)

**Purpose:** Expose all functionality through REST API

**Endpoints:**
- **Chat:** `/chat/message`, `/chat/history`, `/chat/session`
- **Search:** `/search`, `/search/legal`, `/search/suggest`
- **Retrieval:** `/retrieve/question`, `/retrieve/hybrid`, `/retrieve/category`, `/retrieve/advanced`
- **Self-Correction:** `/self-correct`, `/grade/documents`, `/rewrite/query`, `/check/hallucinations`
- **Management:** `/health`, `/status`, `/collection/stats`, `/ingest`

### 5️⃣ User Interface (Chatbot UI)

**Purpose:** Provide intuitive interface for legal document queries

**Features:**
- Modern chat interface with message history
- Real-time typing indicators
- Citation display with source tracking
- Performance metrics visualization
- Session management with local storage
- Settings for customization
- Quick example prompts
- Markdown rendering with syntax highlighting

---

## ✨ Key Features

### 📚 Multi-Modal Document Processing
- **PDF Parsing:** Complex layout handling with structure preservation
- **Image Processing:** Vision model extracts and describes images
- **Text Cleaning:** Legal-specific text normalization
- **Smart Chunking:** Maintains context across splits

### 🔍 Intelligent Retrieval
- **Hybrid Search:** Combines semantic similarity with keyword matching
- **Query Expansion:** Automatically enriches queries with legal terminology
- **Context-Aware:** Understands legal document structure
- **Relevance Filtering:** Removes non-relevant results

### ✅ Quality Assurance
- **Automatic Retry:** Rewrites failed queries and retrieves again
- **Relevance Grading:** LLM-based assessment of document relevance
- **Hallucination Detection:** Validates answers against sources
- **Citation Tracking:** Automatic source attribution

### 🎯 User Experience
- **Modern Chat UI:** Clean, responsive web interface
- **Real-time Feedback:** Typing indicators and loading states
- **Transparent Metrics:** Shows retrieval time, attempts, confidence
- **Session History:** Maintains conversation context
- **Customizable Settings:** Adjust behavior per preference

---

## 🛠️ Technology Stack

### Backend
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | REST API and web server |
| **LLM/Embeddings** | OpenAI (GPT-4, GPT-4V, text-embedding-3-small) | Language understanding and embeddings |
| **Vector DB** | Chroma | Efficient similarity search |
| **Orchestration** | LangChain + LangGraph | RAG workflow and state management |
| **PDF Processing** | Marker, Unstructured.io | Document parsing |
| **Text Processing** | NLTK, spaCy | NLP tasks |
| **Async** | Asyncio, aiohttp | Concurrent processing |

### Frontend
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Markup** | HTML5 | Semantic structure |
| **Styling** | CSS3 | Modern responsive design |
| **Scripting** | JavaScript (Vanilla) | Client-side logic |
| **Markdown** | marked.js | Markdown rendering |
| **Syntax Highlight** | highlight.js | Code block formatting |

### Infrastructure
- **Python 3.11+**
- **Virtual Environment:** venv/conda
- **Package Management:** pip
- **Version Control:** Git
- **Deployment:** Docker, Gunicorn, cloud platforms

---

## 📁 Project Structure

```
multi-modal-legal-system/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration settings
├── app.py                             # FastAPI application (1200+ lines)
│
├── ingestion/                         # Data ingestion modules
│   ├── data.py                        # Main ingestion pipeline
│   ├── pdf_parser.py                  # PDF parsing
│   ├── vision_processor.py            # Image processing
│   ├── text_cleaner.py                # Text normalization
│   ├── chunking.py                    # Document chunking
│   └── vector_store.py                # Vector database management
│
├── retrieval/                         # Retrieval system
│   ├── query_expander.py              # Query expansion with legal ontology
│   ├── retriever.py                   # Hybrid semantic+keyword search
│   └── retrieval_integration.py       # High-level retrieval API
│
├── selfCorrection/                    # Self-correction loop
│   ├── grader.py                      # Document relevance grading
│   └── hallucination_checker.py       # Answer validation
│
├── rewriter.py                        # Query rewriting strategies
├── answer_generator.py                # Answer generation with citations
├── self_correction_orchestrator.py    # LangGraph state machine
│
├── templates/                         # Frontend templates
│   ├── chatbot.html                   # Main chat UI
│   └── index.html                     # Alternative UI
│
├── static/                            # Frontend assets
│   ├── chatbot.js                     # Chat logic
│   ├── chatbot.css                    # Chat styles
│   ├── style.css                      # Alternative styles
│   └── app.js                         # Alternative logic
│
├── chroma_db/                         # Vector database storage
│   └── chroma.sqlite3
│
├── data/                              # Document storage
│   ├── raw/                           # Original documents
│   └── processed/                     # Processed documents
│
└── .env                               # Environment configuration
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- OpenAI API key
- 4GB+ RAM recommended

### Step 1: Clone and Navigate
```bash
cd /Users/mansibisen/Documents/RAG_agent/multi-modal-legal-system
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Packages:**
- `fastapi` - REST API framework
- `langchain` + `langchain-openai` - RAG framework
- `chromadb` - Vector database
- `openai` - LLM and embeddings
- `pydantic` - Data validation
- `pdf2image`, `unstructured` - PDF processing
- `uvicorn` - ASGI server

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
nano .env
```

**Required Variables:**
```
OPENAI_API_KEY=sk-your-key-here
CHROMA_DB_PATH=./chroma_db
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
```

### Step 5: Start the Server
```bash
python app.py
```

Server runs at: `http://localhost:8000`

### Step 6: Access the Chatbot
Open in browser: `http://localhost:8000/chatbot`

---

## 🔄 How It Works

### User Query Flow

```
1. USER SUBMITS QUESTION
   ↓
2. API RECEIVES REQUEST (/chat/message)
   ├─ Session ID created/retrieved
   ├─ Question stored in history
   └─ User message displayed
   ↓
3. QUERY EXPANSION
   ├─ Expand with legal synonyms
   ├─ Add context terms
   └─ Optimize for search
   ↓
4. RETRIEVAL (Attempt 1)
   ├─ Semantic search: vector similarity (OpenAI embeddings)
   ├─ Keyword search: BM25 probabilistic matching
   ├─ Hybrid ranking: combine and rank results
   └─ Return top-k documents
   ↓
5. RELEVANCE GRADING
   ├─ Grade each document (RELEVANT/PARTIAL/IRRELEVANT)
   ├─ Count relevant documents
   └─ Assess coverage
   ↓
6. CHECK IF SUFFICIENT RELEVANT DOCS FOUND
   ├─ IF YES: Go to 8 (Generate Answer)
   └─ IF NO: Go to 7 (Rewrite Query)
   ↓
7. QUERY REWRITING (if needed, max 3 attempts)
   ├─ Analyze failure reason
   ├─ Select rewrite strategy (EXPAND/REPHRASE/FOCUS/etc)
   ├─ Generate improved query
   └─ Return to step 4 with new query
   ↓
8. ANSWER GENERATION
   ├─ Inject relevant document chunks as context
   ├─ Send to LLM with system prompt
   ├─ Extract citations from context
   ├─ Format answer with markdown
   └─ Return initial answer
   ↓
9. HALLUCINATION CHECK
   ├─ Verify claims against source documents
   ├─ Detect fabricated information
   ├─ Assess severity level
   └─ Generate confidence score
   ↓
10. RETURN RESPONSE TO UI
    ├─ Answer text
    ├─ Citations with source references
    ├─ Metrics (attempts, timings, confidence)
    ├─ Sources for verification
    └─ Session ID for history
    ↓
11. UI DISPLAYS RESULT
    ├─ Answer rendered with formatting
    ├─ Citations displayed as clickable references
    ├─ Metrics shown in sidebar
    ├─ Similar questions suggested
    └─ Message added to history
```

### Example: "What are the key clauses in an employment contract?"

**Input:**
```json
{
  "question": "What are the key clauses in an employment contract?",
  "session_id": "session_001"
}
```

**Processing:**
1. **Expansion:** "key clauses employment contract" → "essential provisions, critical terms, standard clauses, employment agreement sections"
2. **Retrieval:** Find 5+ documents mentioning employment contracts
3. **Grading:** 4 relevant, 1 partial → Sufficient (continue)
4. **Generation:** "Employment contracts typically include: Compensation clause, Benefits clause, Termination clause..."
5. **Hallucination Check:** Verify each claim in source documents → All grounded (no fabrication)
6. **Citations:** [1] Section 3.2 Compensation, [2] Section 5.1 Benefits...

**Output:**
```json
{
  "success": true,
  "answer": "Employment contracts typically include...",
  "citations": [
    {"source_id": "Doc 1", "page": 5, "confidence": 0.95}
  ],
  "sources": ["Document excerpt 1", "Document excerpt 2"],
  "metrics": {
    "total_attempts": 1,
    "retrieval_time": 0.234,
    "grading_time": 0.156,
    "generation_time": 0.456,
    "total_time": 0.846
  }
}
```

---

## 📡 API Endpoints

### Chat Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/chat/message` | Send question to AI |
| GET | `/chat/history` | Retrieve session history |
| POST | `/chat/session` | Create new session |
| GET | `/chatbot` | Load chatbot UI |

### Search Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/search` | Basic document search |
| POST | `/search/legal` | Legal-specific search |
| GET | `/search/suggest` | Get search suggestions |

### Retrieval Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/retrieve/question` | Retrieve for questions |
| POST | `/retrieve/hybrid` | Hybrid semantic+keyword |
| POST | `/retrieve/category` | Search by category |
| POST | `/retrieve/compare` | Compare clauses |
| POST | `/retrieve/related` | Find related clauses |
| POST | `/retrieve/advanced` | Advanced search |

### Self-Correction Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/self-correct` | Run full workflow |
| POST | `/grade/documents` | Grade relevance |
| POST | `/rewrite/query` | Rewrite query |
| POST | `/check/hallucinations` | Check for hallucinations |
| POST | `/generate/answer` | Generate answer |

### Management Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/status` | System status |
| GET | `/collection/stats` | Collection statistics |
| POST | `/collection/clear` | Clear all documents |
| POST | `/ingest` | Ingest documents |
| GET | `/` | API documentation |

---

## 💡 Usage Examples

### Example 1: Simple Legal Question
```bash
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a force majeure clause?",
    "session_id": "session_001"
  }'
```

**Response:** Comprehensive explanation with citations

### Example 2: Hybrid Search
```bash
curl -X POST "http://localhost:8000/retrieve/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find all limitation of liability clauses",
    "k": 5
  }'
```

**Response:** Top 5 documents combined by semantic + keyword relevance

### Example 3: Grade Documents
```bash
curl -X POST "http://localhost:8000/grade/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are non-compete clause requirements?",
    "documents": ["Document text 1...", "Document text 2..."]
  }'
```

**Response:** Relevance grades for each document

---

## 📊 Performance & Metrics

### Typical Response Times
- Query Expansion: 50-150ms
- Retrieval (Hybrid Search): 200-500ms
- Document Grading: 300-800ms
- Answer Generation: 800-2000ms
- Hallucination Check: 500-1500ms
- **Total:** 2-5 seconds (single attempt)

### Optimization Strategies
1. **Vector Store:** Efficient similarity search with indexing
2. **Batch Processing:** Process multiple images concurrently
3. **Async Operations:** Parallel retrieval and grading
4. **Caching:** Browser-based session caching
5. **Smart Retry:** Minimizes unnecessary API calls

### Metrics Tracking
Every response includes:
- `total_attempts`: Number of retrieval attempts
- `retrieval_time`: Document search duration
- `grading_time`: Relevance assessment duration
- `rewriting_time`: Query modification duration
- `generation_time`: Answer synthesis duration
- `total_time`: End-to-end duration

---

## 🚀 Deployment

### Development
```bash
python app.py
# Runs with auto-reload on port 8000
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  app:app --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
CHROMA_DB_PATH=/var/lib/legal_ai/chroma_db
LOG_LEVEL=INFO
MAX_WORKERS=4
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:**
```bash
source myenv/bin/activate
pip install -r requirements.txt
```

### Issue: OpenAI API errors
**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
# Verify in .env file
```

### Issue: Slow responses
**Solution:**
- Reduce `k` parameter in searches
- Check vector store size
- Optimize chunk size in config.py

### Issue: Poor retrieval results
**Solution:**
- Index more documents
- Adjust similarity threshold
- Review query expansion patterns
- Check document quality

### Issue: Port 8000 already in use
**Solution:**
```bash
python -m uvicorn app:app --port 8001
```

---

## 📈 Future Enhancements

1. **Real-time Streaming:** Stream responses as generated
2. **Voice Input/Output:** Voice-based interaction
3. **Custom Models:** Support fine-tuned legal LLMs
4. **Advanced Analytics:** Usage insights and optimization
5. **Collaboration:** Share and discuss documents with teams
6. **Integration Hub:** Connect with external legal tools
7. **Mobile App:** Native iOS/Android applications
8. **Document Upload:** Direct file upload and processing

---

## 📝 Configuration

Edit `config.py` to customize:

```python
# Models
EMBEDDING_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4o-mini"
LLM_MODEL = "gpt-4-turbo"

# Chunking
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
SECTION_AWARE_CHUNKING = True

# Vector Store
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_documents"

# Retry Logic
MAX_RETRIES = 3
RETRY_TIMEOUT = 30
```

---

## 🔒 Security Considerations

- ✅ Validate all user inputs
- ✅ Use environment variables for API keys
- ✅ Implement rate limiting on production
- ✅ Add authentication if needed
- ✅ Use HTTPS in production
- ✅ Monitor API usage and costs
- ✅ Log sensitive operations

---

## 📞 Support & Documentation

- **API Docs:** `/docs` (Swagger UI)
- **Alternative Docs:** `/redoc` (ReDoc)
- **Health Check:** `/health`
- **System Status:** `/status`

---

## 📄 License

This project is part of the Multi-Modal Legal System initiative.

---

## 🎉 Summary

You now have a **production-ready AI-powered legal chatbot system** that:
- ✅ Intelligently processes multi-modal legal documents
- ✅ Performs hybrid semantic + keyword retrieval
- ✅ Self-corrects through automatic query rewriting
- ✅ Validates answers against sources
- ✅ Provides transparent metrics and citations
- ✅ Offers modern, responsive user interface
- ✅ Scales to handle thousands of documents
- ✅ Integrates seamlessly with existing workflows

**Start exploring legal documents with AI today!**

```bash
# Open in browser
open http://localhost:8000/chatbot
```

---

**Last Updated:** April 18, 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅
