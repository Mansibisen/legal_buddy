# Multi-Modal Legal AI Chatbot System - Design Document

**Project:** Multi-Modal Legal AI Chatbot System  
**Version:** 1.0.0  
**Date:** May 2026  
**Author:** Legal Buddy Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [High-Level Architecture](#high-level-architecture)
4. [Low-Level Architecture](#low-level-architecture)
5. [Core Concepts & Technologies](#core-concepts--technologies)
6. [Detailed Component Design](#detailed-component-design)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Design Patterns](#design-patterns)
9. [Technology Stack](#technology-stack)
10. [Key Algorithms & Techniques](#key-algorithms--techniques)

---

## Executive Summary

The **Multi-Modal Legal AI Chatbot System** is a sophisticated enterprise-grade AI solution designed to intelligently process, understand, and answer complex legal questions by analyzing legal documents, contracts, and case materials. The system combines advanced NLP techniques, vector-based semantic search, and self-correction mechanisms to provide accurate, cited, and verifiable legal information.

### Key Capabilities:
- **Multi-Modal Processing**: Handles text and images from PDFs
- **Hybrid Retrieval**: Combines semantic search with keyword matching
- **Self-Correcting**: Validates and refines answers automatically
- **Hallucination Detection**: Prevents AI from fabricating information
- **Citation Support**: Provides source references for all answers
- **Production-Ready**: REST API, logging, and monitoring built-in

---

## System Overview

### Problem Statement
Legal professionals face critical challenges:
- **Time-Consuming Reviews**: Manual document analysis takes days/weeks
- **Information Loss**: Missing critical clauses across large datasets
- **Consistency Issues**: Different interpretations of same content
- **Risk Management**: Overlooking important legal provisions
- **Scalability Problems**: Cannot handle large document volumes

### Solution Architecture
The system implements a **Retrieval-Augmented Generation (RAG) pipeline with self-correction** to ensure:
- **Accuracy**: Retrieved documents ground all answers
- **Relevance**: Multi-stage filtering ensures quality results
- **Transparency**: Complete citation trail for verification
- **Reliability**: Automatic hallucination detection and answer refinement

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                         │
│         Web-based Chat Interface (Streamlit/FastAPI UI)          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                            │
│                  FastAPI REST API Endpoints                       │
│  • POST /chat/message       • GET /chat/history                  │
│  • POST /chat/session       • GET /chatbot                       │
│  • POST /documents/ingest   • GET /documents                     │
│  • POST /search             • DELETE /documents/{id}             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│   RETRIEVAL    │ │  SELF-CORRECT  │ │     ANSWER     │
│   PIPELINE     │ │    LOOP        │ │   GENERATION   │
└────────────────┘ └────────────────┘ └────────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                            │
│  PDF Parser → Image Extractor → Text Cleaner → Vision → Chunk  │
│                                                     │              │
│                                                     ▼              │
│                                          Embedding Generation     │
└──────────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  VECTOR DB     │ │   METADATA     │ │  FILE STORAGE  │
│  (Chroma)      │ │   (SQLITE)     │ │  (PDF cache)   │
└────────────────┘ └────────────────┘ └────────────────┘
```

### Core Layers Explained

#### 1. **User Interface Layer**
- **Purpose**: Enables interaction with the system
- **Implementation**: Streamlit web app + FastAPI REST API
- **Responsibilities**: Session management, chat history, user feedback

#### 2. **API Gateway Layer**
- **Purpose**: Exposes system functionality as REST endpoints
- **Framework**: FastAPI with CORS support
- **Features**: Request validation, error handling, response formatting

#### 3. **Processing Pipeline**
Composed of three interacting modules:

**a) Retrieval Pipeline**
- Searches vector database using hybrid methods
- Expands queries using domain knowledge
- Ranks results by relevance

**b) Self-Correction Loop**
- Grades retrieved documents
- Rewrites failing queries
- Validates answers for hallucinations

**c) Answer Generation**
- Synthesizes coherent responses
- Generates citations
- Formats with supporting media

#### 4. **Ingestion Pipeline**
- Transforms raw PDFs into searchable knowledge
- Multi-stage processing with quality checks
- Stores embeddings and metadata

#### 5. **Data Storage**
- **Vector DB (Chroma)**: Semantic search index
- **Metadata Store (SQLite)**: Document metadata
- **File Cache**: Original PDFs for reference

---

## Low-Level Architecture

### Component Decomposition

```
INGESTION SUBSYSTEM
├── data.py
│   ├── MultiModalLegalIngestionPipeline (Orchestrator)
│   ├── IngestionStats (Data Container)
│   └── Batch Processing Logic
├── pdf_parser.py
│   ├── PDFParser (Main Parser)
│   ├── ImageExtractor (Extract images from PDFs)
│   └── DocumentMetadata (Store document info)
├── vision_processor.py
│   ├── VisionProcessor (OCR + understanding)
│   ├── ImageAnalyzer (Semantic image understanding)
│   └── ImageCache (Store processed images)
├── text_cleaner.py
│   ├── TextCleaner (Normalize and clean text)
│   ├── BoilerplateRemover (Remove headers/footers)
│   └── LanguageNormalizer
├── chunking.py
│   ├── HybridChunker (Smart chunking)
│   ├── SemanticChunker (Context-aware splits)
│   ├── SectionAwareChunker (Legal section awareness)
│   └── ChunkMetadata
└── vector_store.py
    ├── VectorStore (Chroma wrapper)
    ├── EmbeddingManager (Generate/retrieve embeddings)
    └── IndexManager

RETRIEVAL SUBSYSTEM
├── retriever.py
│   ├── HybridRetriever (Semantic + Keyword)
│   ├── SemanticRetriever (Vector similarity)
│   ├── KeywordRetriever (BM25 search)
│   ├── ScoreAggregator (Combine scores)
│   └── RetrievedDocument (Result container)
├── retrieval_integration.py
│   ├── LegalRetrievalEngine (Main orchestrator)
│   ├── QueryPreprocessor
│   ├── DocumentRanker
│   └── ContextBuilder
└── query_expander.py
    ├── QueryExpander (Add legal synonyms)
    ├── QueryOptimizer (Optimize for search)
    ├── LegalOntology (Domain knowledge)
    └── QueryType Classifier

SELF-CORRECTION SUBSYSTEM
├── grader.py
│   ├── DocumentGrader (Grade relevance)
│   ├── RelevanceEvaluator (LLM-based evaluation)
│   └── KeywordFallback (Keyword-based grading)
├── rewriter.py
│   ├── QueryRewriter (Improve queries)
│   ├── RewriteStrategy (Different strategies)
│   └── PromptBuilder
├── hallucination_checker.py
│   ├── HallucinationChecker (Detect false info)
│   ├── FactValidator (Ground in source)
│   └── ConfidenceScorer
└── self_correction_orchestrator.py
    ├── SelfCorrectionOrchestrator (Main workflow)
    ├── WorkflowState (State machine)
    ├── LangGraph Integration
    └── Transition Logic

ANSWER GENERATION SUBSYSTEM
├── answer_generator.py
│   ├── MultiModalAnswerGenerator (Generate answers)
│   ├── PromptBuilder (Craft LLM prompts)
│   ├── Citation (Citation data class)
│   ├── AnswerFormatter (Format output)
│   └── MultiModalAnswer (Container)
└── ollama_integration
    ├── Model Loader
    ├── Response Parser
    └── Error Handling

CORE INFRASTRUCTURE
├── config.py
│   ├── Settings (Pydantic config)
│   └── Environment variables
├── app.py
│   ├── FastAPI Application
│   ├── Endpoint Handlers
│   ├── Request/Response Models
│   └── Middleware (CORS, logging)
└── logging_config
    ├── Rotating File Handlers
    ├── Console Handlers
    └── Error Tracking
```

### Module Responsibilities

| Module | Purpose | Key Classes | Dependencies |
|--------|---------|------------|--------------|
| **pdf_parser.py** | Extract text & images from PDFs | PDFParser, ImageExtractor | pypdf2, pdf2image, pytesseract |
| **vision_processor.py** | Process images with vision model | VisionProcessor, ImageAnalyzer | ollama, llava model |
| **text_cleaner.py** | Normalize and clean document text | TextCleaner, BoilerplateRemover | regex, beautifulsoup |
| **chunking.py** | Smart document segmentation | HybridChunker, SemanticChunker | langchain-text-splitters |
| **vector_store.py** | Manage embeddings and search index | VectorStore, EmbeddingManager | chromadb, ollama |
| **retriever.py** | Hybrid semantic + keyword search | HybridRetriever, KeywordRetriever | chromadb, numpy |
| **query_expander.py** | Expand queries with legal synonyms | QueryExpander, LegalOntology | custom |
| **grader.py** | Grade document relevance | DocumentGrader, RelevanceEvaluator | ollama, llama2 model |
| **rewriter.py** | Rewrite failed queries | QueryRewriter | ollama, llama2 model |
| **hallucination_checker.py** | Detect false information | HallucinationChecker | ollama, llama2 model |
| **self_correction_orchestrator.py** | Orchestrate correction loop | SelfCorrectionOrchestrator | langgraph |
| **answer_generator.py** | Generate final answers | MultiModalAnswerGenerator | ollama, llama2 model |
| **app.py** | REST API and orchestration | FastAPI application | fastapi, uvicorn |

---

## Core Concepts & Technologies

### 1. **Retrieval-Augmented Generation (RAG)**

**Concept**: Combine retrieval and generation to ground answers in source documents.

**Implementation**:
```
User Query → Query Expansion → Hybrid Retrieval → Relevant Docs 
→ LLM Prompt with Context → Generated Answer → Validation
```

**Benefits**:
- Answers are grounded in actual documents
- Reduced hallucinations
- Can provide citations
- Factually accurate

### 2. **Hybrid Search Strategy**

**Semantic Search** (Vector Similarity)
- Transforms query and documents into embeddings
- Finds semantically similar content even with different words
- Uses cosine similarity for ranking
- Model: `all-minilm` embedder

**Keyword Search** (BM25)
- Traditional information retrieval method
- Exact word matching with frequency weighting
- Good for domain-specific terminology
- Complements semantic search

**Score Aggregation**:
```python
final_score = α × semantic_score + (1-α) × keyword_score
# Typically α = 0.7 for 70% semantic, 30% keyword
```

### 3. **Self-Correction Loop (Multi-Agent)**

**State Machine Workflow**:
```
START → RETRIEVE → GRADE → REWRITE → RETRIEVE_AGAIN → GENERATE 
→ CHECK_HALLUCINATION → (REFINE or COMPLETE)
```

**Agents**:

| Agent | Role | Decision |
|-------|------|----------|
| **Document Grader** | Evaluates relevance | Keep, improve, or retry retrieval |
| **Query Rewriter** | Improves failed queries | Expand, simplify, rephrase, focus, decompose |
| **Hallucination Checker** | Validates answers | Accept or refine |
| **Answer Refiner** | Fixes hallucinations | Regenerate with constraints |

**Loop Benefits**:
- Autonomous quality improvement
- Multiple retry attempts
- Guaranteed minimal relevance threshold
- Transparency on corrections made

### 4. **Intelligent Chunking**

**Problem**: Large documents must be split into indexed chunks, but arbitrary splits lose context.

**Solutions**:

**a) Overlap Strategy**
- Chunk size: 1024 characters
- Overlap: 200 characters
- Maintains context across chunks

**b) Section-Aware Chunking**
- Recognizes legal sections
- Keeps clauses intact
- Preserves logical structure

**c) Semantic Boundaries**
- Uses sentence boundaries
- Avoids mid-sentence splits
- Maintains coherence

### 5. **Legal Domain Ontology**

**Purpose**: Enhance retrieval for legal terminology

**Structure**:
```python
SYNONYMS = {
    "payment": ["compensation", "fee", "cost", "remittance", ...],
    "breach": ["violation", "infringement", "default", ...],
    "liable": ["responsible", "accountable", "bound", ...],
    "terminate": ["cancel", "end", "dissolve", ...],
    # ... 30+ legal terms with expansions
}

QUERY_TYPES = [
    "payment", "liability", "confidentiality", 
    "termination", "IP", "obligations", "warranties", ...
]
```

**Usage**: Query classification + synonym expansion before retrieval

### 6. **Multi-Modal Processing**

**PDF Level**:
- Extract text with `pypdf`
- Extract images with `pdf2image`
- OCR via `pytesseract`

**Image Understanding**:
- Use local `llava` vision model
- Analyze charts, diagrams, signatures
- Extract text from images (OCR)
- Generate image descriptions

**Integration**:
- Images become part of answer context
- Visual evidence supports text answers
- Important diagrams highlighted in response

### 7. **Vector Database Architecture**

**Chroma DB Features**:
- Lightweight, embedded vector database
- Built-in embedding management
- Filtering by metadata
- Similarity search

**Schema**:
```
Collection: legal_documents
├── Documents (text chunks)
├── Embeddings (vector representations)
├── Metadatas
│   ├── document_id
│   ├── chunk_id
│   ├── page_number
│   ├── section_title
│   ├── chunk_type (text/image)
│   └── source_file
└── IDs (unique per chunk)
```

### 8. **LangChain Integration**

**Components Used**:
- **langchain-core**: Base abstractions
- **langchain-ollama**: Ollama LLM integration
- **langchain-chroma**: Vector store integration
- **langchain-text-splitters**: Smart text chunking
- **langgraph**: State machine for workflows

**Benefits**:
- Standardized abstractions
- Easy model switching
- Built-in memory/history management
- Prompt templating

### 9. **LangGraph State Machine**

**Purpose**: Orchestrate complex multi-step workflows with state management

**Key Components**:
```python
StateGraph(state_schema=SelfCorrectionWorkflowState)
├── Nodes (Agent functions)
│   ├── retrieve_documents
│   ├── grade_documents
│   ├── rewrite_query
│   ├── generate_answer
│   ├── check_hallucination
│   └── refine_answer
└── Edges (Transitions)
    ├── Conditional edges (based on grades/checks)
    └── Fallback edges (for failures)
```

**State Persistence**:
- Maintains entire workflow state
- Tracks document scores
- Records all rewrites
- Logs all decisions

### 10. **Ollama Local LLM Integration**

**Why Local Models?**:
- Privacy: Data never leaves premises
- Cost: No API charges
- Latency: No network roundtrip
- Control: Full customization

**Models Used**:
- **llama2** (7B parameters) - Main reasoning
- **llava** (13B parameters) - Vision understanding
- **all-minilm** - Fast embeddings

**Configuration**:
```python
OllamaLLM(
    model="llama2",
    base_url="http://localhost:11434",
    temperature=0.3,  # Deterministic
    num_predict=2048  # Max tokens
)
```

---

## Detailed Component Design

### 1. Ingestion Pipeline (`ingestion/data.py`)

**Class: `MultiModalLegalIngestionPipeline`**

```python
def ingest_document(pdf_path: str, document_id: str) -> IngestionStats:
    """
    End-to-end document ingestion
    
    Steps:
    1. Parse PDF (text + images)
    2. Extract and process images with vision
    3. Clean text
    4. Intelligently chunk document
    5. Generate embeddings
    6. Store in vector DB
    
    Returns: Statistics and metadata
    """
```

**Processing Steps**:

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1 | PDF file | pypdf2, pdf2image | Text, images, page info |
| 2 | Images | Vision model | Image descriptions, OCR text |
| 3 | Raw text | Regex, NLP | Clean, normalized text |
| 4 | Clean text | Semantic chunker | ~512 chunks per document |
| 5 | Chunks | Ollama embedder | 384-dim vectors |
| 6 | Chunks+embeddings | Chroma | Indexed, searchable data |

**Error Handling**:
- Corrupted PDFs → Skip with logging
- Vision model fails → Continue with text only
- Embedding fails → Fallback to BM25 only
- Storage fails → Retry with backoff

### 2. Retriever (`retriever/retriever.py`)

**Class: `HybridRetriever`**

**Semantic Search**:
```python
def semantic_search(query: str, k: int = 5) -> List[RetrievedDocument]:
    # 1. Embed query
    query_embedding = embed(query)
    
    # 2. Search vector DB
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 3. Score by similarity (0-1)
    return results_with_scores
```

**Keyword Search**:
```python
def keyword_search(query: str, k: int = 5) -> List[RetrievedDocument]:
    # 1. Tokenize query
    tokens = tokenize(query)
    
    # 2. BM25 scoring
    scores = {}
    for doc in all_documents:
        score = calculate_bm25(tokens, doc)
        scores[doc] = score
    
    # 3. Return top-k
    return sorted(scores.items(), reverse=True)[:k]
```

**Hybrid Combination**:
```python
def hybrid_search(query: str, k: int = 5, alpha: float = 0.7) -> List[RetrievedDocument]:
    semantic_results = semantic_search(query, k)
    keyword_results = keyword_search(query, k)
    
    # Normalize scores to [0, 1]
    sem_norm = normalize(semantic_results)
    kw_norm = normalize(keyword_results)
    
    # Combine with weights
    final_scores = {}
    for doc in combined_results:
        score = alpha * sem_norm[doc] + (1-alpha) * kw_norm[doc]
        final_scores[doc] = score
    
    return sorted(final_scores.items(), reverse=True)[:k]
```

### 3. Document Grader (`selfCorrection/grader.py`)

**Purpose**: Validate that retrieved documents answer the question

**Logic**:
```python
def grade_document(question: str, document: str) -> GradeScore:
    """
    LLM Prompt:
    ---
    You are a legal expert evaluator.
    Question: {question}
    Document: {document}
    
    Does this document answer the question?
    1. RELEVANT: Directly answers the question
    2. PARTIALLY_RELEVANT: Partially addresses question
    3. IRRELEVANT: Doesn't help answer question
    
    Respond with: [RELEVANT|PARTIALLY|IRRELEVANT]
    ---
    
    Parse response and return grade
    """
```

**Fallback Strategy**:
If LLM unavailable, use keyword matching:
```python
def grade_by_keywords(question: str, document: str) -> GradeScore:
    q_keywords = extract_keywords(question)
    doc_keywords = extract_keywords(document)
    
    overlap = len(q_keywords & doc_keywords)
    coverage = overlap / len(q_keywords)
    
    if coverage > 0.7:
        return GradeScore.RELEVANT
    elif coverage > 0.4:
        return GradeScore.PARTIALLY_RELEVANT
    else:
        return GradeScore.IRRELEVANT
```

### 4. Query Rewriter (`selfCorrection/rewriter.py`)

**Strategies**:

| Strategy | When | Example |
|----------|------|---------|
| **EXPAND** | Too narrow | "payment terms" → "payment terms AND late fees AND interest" |
| **SIMPLIFY** | Over-complex | Remove modifiers, focus on core |
| **REPHRASE** | Different angles | "Who is liable?" → "Liability clause" |
| **FOCUS** | Too broad | Remove irrelevant modifiers |
| **DECOMPOSE** | Complex query | Multi-part breakdown |

### 5. Hallucination Checker (`selfCorrection/hallucination_checker.py`)

**Methodology**:
```python
def check_hallucination(question: str, answer: str, sources: List[str]) -> HallucinationLevel:
    """
    LLM Prompt:
    ---
    Question: {question}
    Answer: {answer}
    Source Documents: {sources}
    
    Check if answer claims are:
    1. Supported by sources (grounded)
    2. Contradicted by sources (hallucinated)
    3. Not verifiable from sources (unverifiable)
    
    For each claim in the answer:
    - Cite supporting evidence OR
    - Flag as hallucination OR
    - Mark as unverifiable
    
    Return:
    {
        "hallucination_level": NONE|MINOR|MODERATE|SEVERE,
        "hallucinated_claims": [...],
        "grounded_claims": [...],
        "unverifiable_claims": [...]
    }
    ---
    """
```

### 6. Answer Generator (`answer_generator.py`)

**Architecture**:

```python
def generate_answer(
    question: str,
    context: List[str],
    images: List[Image] = None
) -> MultiModalAnswer:
    """
    1. Build comprehensive prompt
    2. Include retrieved documents as context
    3. Include relevant images
    4. Ask for structured output (answer + citations)
    5. Parse response
    6. Extract citations
    7. Format as MultiModalAnswer
    """
    
    prompt = build_prompt(question, context, images)
    raw_response = llm.invoke(prompt)
    
    answer_text = extract_answer(raw_response)
    citations = extract_citations(raw_response, context)
    
    return MultiModalAnswer(
        answer_text=answer_text,
        citations=citations,
        source_documents=context,
        supporting_images=images,
        confidence=calculate_confidence(answer_text, context)
    )
```

**Prompt Template**:
```
You are an expert legal advisor. Answer the following question based 
ONLY on the provided context. If the context doesn't contain the answer, 
say "I cannot answer based on provided documents."

Question: {question}

Context from legal documents:
{context}

Supporting images/diagrams:
{image_descriptions}

Provide:
1. Direct answer to the question
2. Relevant citations [cite: doc_id, page X]
3. Key legal references
4. Confidence level (high/medium/low)
```

### 7. Self-Correction Orchestrator (`selfCorrection/self_correction_orchestrator.py`)

**Workflow State Machine**:

```
         ┌─────────────────┐
         │     START       │
         └────────┬────────┘
                  │
                  ▼
           ┌──────────────┐
           │   RETRIEVE   │ (Get initial docs)
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │    GRADE     │ (Evaluate relevance)
           └──────┬───────┘
                  │
         ┌────────┴────────┐
         │                 │
      Relevant?          No
         │                 │
      Yes│            Retry count
         │            exceeded?
         │                 │
         │            Yes──┼──→ GENERATE
         │                 │    (Use what we have)
         │                 │
         │                 ▼
         │            REWRITE (Improve query)
         │                 │
         │                 ▼
         │            RETRIEVE_AGAIN
         │                 │
         │                 └──→ (back to GRADE)
         │
         ▼
    ┌──────────────┐
    │  GENERATE    │ (Create answer)
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────┐
    │ CHECK_HALLUCINATION  │
    └──────┬───────────────┘
           │
        Has Hallucination?
           │
      Yes  │  No
          │  └─────────────┐
          │                │
          ▼                ▼
       REFINE         COMPLETE
        (Fix)         (Return)
          │                │
          └────────┬───────┘
                   ▼
              ┌────────┐
              │  END   │
              └────────┘
```

**State Transitions**:
```python
graph = StateGraph(SelfCorrectionWorkflowState)

# Add nodes (agent functions)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("grade", grade_documents)
graph.add_node("rewrite", rewrite_query)
graph.add_node("generate", generate_answer)
graph.add_node("check_hallucination", check_hallucination)

# Add edges (transitions)
graph.add_edge("retrieve", "grade")

# Conditional edges (decisions)
graph.add_conditional_edges(
    "grade",
    lambda state: "generate" if state.relevant_count > 2 else "rewrite",
    {"generate": "generate", "rewrite": "rewrite"}
)

graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", "check_hallucination")

graph.add_conditional_edges(
    "check_hallucination",
    lambda state: "refine" if state.hallucination_level else "complete",
    {"refine": "refine", "complete": "complete"}
)
```

---

## Data Flow Diagrams

### Flow 1: Document Ingestion

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  1. PDF Parser (pypdf2)            │
    │  - Extract pages                   │
    │  - Extract text                    │
    │  - Get page dimensions             │
    ├─────────────────────────────────────┤
    │ Output: text, page_count, metadata │
    └────────┬────────────────────────────┘
             │
    ┌────────┴──────────┐
    │                   │
    ▼                   ▼
 ┌──────────────┐   ┌──────────────────────┐
 │  PDF2Image   │   │ Text Cleaning        │
 │  Extract     │   │ - Remove boilerplate │
 │  images      │   │ - Normalize spaces   │
 │              │   │ - Fix encoding       │
 └──────┬───────┘   └──────┬───────────────┘
        │                  │
        ▼                  │
    ┌──────────────┐      │
    │  Vision      │      │
    │  Model       │      │
    │  (llava)     │      │
    │  - OCR       │      │
    │  - Describe  │      │
    └──────┬───────┘      │
           │              │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │   Chunking   │
           │ - Split text │
           │ - Overlap    │
           │ - Metadata   │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │  Embeddings  │
           │ (all-minilm) │
           │ 384-dim vec  │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │  Chroma DB   │
           │  Indexed     │
           └──────────────┘
```

### Flow 2: Query Processing & Answer Generation

```
┌──────────────────┐
│  User Question   │
└────────┬─────────┘
         │
         ▼
    ┌────────────────────────────────────┐
    │  Query Expansion                  │
    │ - Classify query type             │
    │ - Add legal synonyms              │
    │ - Optimize for search             │
    ├────────────────────────────────────┤
    │ Input: "payment terms"            │
    │ Output: "payment terms OR         │
    │          compensation clauses OR   │
    │          fee structures..."       │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │  Hybrid Retrieval                 │
    │ - Semantic search                 │
    │ - Keyword search                  │
    │ - Score aggregation               │
    │ - Top-5 documents                 │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │  Document Grading (LOOP)          │
    │ - LLM evaluates each doc          │
    │ - Scores relevance                │
    │ - If relevant_count >= 2, continue│
    │ - Else, rewrite query             │
    │ - Retry up to 3 times             │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │  Answer Generation                │
    │ - Build prompt with context       │
    │ - Include retrieved docs          │
    │ - Include images                  │
    │ - Ask for citations               │
    │ - Parse LLM response              │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │  Hallucination Check              │
    │ - LLM validates claims            │
    │ - Checks grounding                │
    │ - If hallucinating, refine        │
    │ - Regenerate if severe            │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │  Response Formatting              │
    │ - Add citations                   │
    │ - Add confidence scores           │
    │ - Include source docs             │
    │ - Add supporting images           │
    └────────┬───────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  User Response   │
    │ - Answer text    │
    │ - Citations      │
    │ - Sources        │
    │ - Images         │
    │ - Confidence     │
    └──────────────────┘
```

---

## Design Patterns

### 1. **Pipeline Pattern**
Used in data ingestion to chain transformations:
```
Input → Parser → Cleaner → Chunker → Embedder → Storage
```
Each stage is independent and testable.

### 2. **Strategy Pattern**
Used for different retrieval and rewriting strategies:
```python
class RetrievalStrategy(ABC):
    def search(self, query: str) -> List[Document]:
        pass

class SemanticStrategy(RetrievalStrategy):
    def search(self, query: str):
        # Vector DB search

class KeywordStrategy(RetrievalStrategy):
    def search(self, query: str):
        # BM25 search

# Runtime selection
strategy = semantic_strategy if use_semantic else keyword_strategy
results = strategy.search(query)
```

### 3. **Adapter Pattern**
Used to integrate Ollama into LangChain:
```python
from langchain_ollama import OllamaLLM

# Adapter provides standard LLM interface
llm = OllamaLLM(model="llama2", base_url=settings.OLLAMA_BASE_URL)

# Works with any LangChain component
response = llm.invoke(prompt)
```

### 4. **State Machine Pattern**
Used for self-correction workflow:
```python
class WorkflowState:
    question: str
    retrieved_documents: List[str]
    grade_assessment: Dict[str, Any]
    hallucination_level: HallucinationLevel
    # ... more state
    
graph = StateGraph(state_schema=WorkflowState)
# Define states and transitions
```

### 5. **Wrapper/Facade Pattern**
Used to simplify complex subsystems:
```python
# Complex retrieval logic hidden
class LegalRetrievalEngine:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.expander = QueryExpander()
        self.ranker = DocumentRanker()
    
    def retrieve(self, query: str) -> List[Document]:
        # Client calls single method
        # Complex orchestration happens internally
```

### 6. **Factory Pattern**
Used for creating different components:
```python
class ChunkerFactory:
    @staticmethod
    def create_chunker(strategy: str) -> Chunker:
        if strategy == "hybrid":
            return HybridChunker()
        elif strategy == "semantic":
            return SemanticChunker()
        else:
            return SimpleChunker()

chunker = ChunkerFactory.create_chunker("hybrid")
```

### 7. **Observer Pattern**
Used for logging and monitoring:
```python
# Pipeline components log events
parser.add_listener(logger)
chunker.add_listener(metrics_collector)

# Listeners react to events
logger.on_event("chunk_created", lambda e: log(e))
```

### 8. **Caching Strategy**
Used to optimize repeated operations:
```python
class VectorStore:
    def __init__(self):
        self.embedding_cache = {}
        self.search_cache = {}
    
    def embed_with_cache(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        embedding = self.generate_embedding(text)
        self.embedding_cache[text] = embedding
        return embedding
```

---

## Technology Stack

### Core Framework & API
| Technology | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | 0.109+ | REST API framework |
| **Uvicorn** | 0.27+ | ASGI server |
| **Pydantic** | 2.0+ | Request validation |
| **Streamlit** | 1.29+ | Web UI |

### LLM & NLP
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Ollama** | Latest | Local LLM runtime |
| **LangChain** | 0.1+ | LLM orchestration |
| **LangGraph** | 0.1+ | State machine workflows |
| **llama2** | 7B | Main reasoning model |
| **llava** | 13B | Vision model |
| **all-minilm** | Latest | Embedding model |

### Document Processing
| Technology | Version | Purpose |
|-----------|---------|---------|
| **PyPDF** | 3.0+ | PDF text extraction |
| **pdf2image** | 1.16+ | PDF to images |
| **PyTesseract** | 0.3+ | OCR |
| **BeautifulSoup4** | 4.12+ | HTML/XML parsing |
| **python-docx** | 0.8+ | DOCX support |
| **OpenCV** | 4.8+ | Image processing |

### Data Storage
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Chroma** | 0.3+ | Vector database |
| **SQLite** | 3.40+ | Metadata storage |

### Configuration & Environment
| Technology | Version | Purpose |
|-----------|---------|---------|
| **pydantic-settings** | 2.0+ | Config management |
| **python-dotenv** | 1.0+ | Environment variables |

---

## Key Algorithms & Techniques

### 1. **BM25 (Best Matching 25)**

**Purpose**: Probabilistic keyword ranking algorithm

**Formula**:
$$\text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $q_i$ = query term
- $D$ = document
- $f(q_i, D)$ = term frequency in document
- $|D|$ = document length
- $\text{avgdl}$ = average document length
- $k_1$, $b$ = tuning parameters (typically 1.5, 0.75)
- $\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$

**Implementation**: Used as fallback when vector search unavailable

### 2. **Cosine Similarity**

**Purpose**: Measure semantic similarity between embeddings

**Formula**:
$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \cdot |\vec{B}|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Properties**:
- Range: [-1, 1]
- 1 = identical direction
- 0 = orthogonal
- -1 = opposite direction

**Application**: Ranking vector DB search results

### 3. **Weighted Score Aggregation**

**Purpose**: Combine multiple ranking signals

**Formula**:
$$\text{Score}_{\text{final}} = \alpha \cdot \text{Score}_{\text{semantic}} + (1-\alpha) \cdot \text{Score}_{\text{keyword}}$$

**Tuning**:
- $\alpha = 0.7$ (default): 70% semantic, 30% keyword
- $\alpha = 1.0$: Pure semantic (for dense text)
- $\alpha = 0.5$: Equal weighting
- $\alpha = 0.3$: Keyword-heavy (for domain-specific terms)

### 4. **Inverse Document Frequency (IDF)**

**Purpose**: Weigh term importance across corpus

**Formula**:
$$\text{IDF}(term) = \log \frac{\text{Total Documents}}{\text{Documents Containing Term}}$$

**Interpretation**:
- Common terms (e.g., "the") → low IDF
- Rare terms (e.g., "lien") → high IDF
- Unique terms get higher weight in ranking

### 5. **Sliding Window with Overlap**

**Purpose**: Chunk documents while preserving context

**Algorithm**:
```
1. Split text into sentences
2. Create chunks of size S characters
3. Overlap by O characters (typically O = S/5)
4. Adjust boundaries to sentence breaks
5. Add metadata (page, section)
```

**Example**:
```
Text: "Clause 1 is about payment. Clause 2 is about liability. Clause 3 is about confidentiality."

Chunk 1: "Clause 1 is about payment. Clause 2"
Chunk 2: "Clause 2 is about liability. Clause 3"
Chunk 3: "Clause 3 is about confidentiality."
```

### 6. **Semantic Sentence Splitting**

**Purpose**: Split on meaning boundaries, not character limits

**Algorithm**:
1. Identify sentence boundaries
2. Group sentences by topic coherence
3. Merge/split to meet size constraints
4. Preserve clause structure
5. Maintain legal semantics

### 7. **Query Expansion with Legal Ontology**

**Purpose**: Augment query with domain-specific synonyms

**Algorithm**:
```
1. Extract entities from query
2. Classify query type (payment, liability, etc.)
3. Look up synonyms in ontology
4. Expand query with 2-3 key synonyms
5. Combine with AND/OR operators
6. Rerank original + expansion results
```

**Example**:
```
Input:  "payment terms"
Expand: "payment terms" OR "compensation clauses" OR "fee structures"
Result: Retrieved more relevant documents
```

### 8. **Multi-Agent Self-Correction Loop**

**Purpose**: Iteratively improve retrieval and answer quality

**Algorithm**:
```
retry_count = 0
max_retries = 3

while retry_count < max_retries:
    # Retrieve documents
    docs = retrieve(query)
    
    # Grade relevance
    grades = grade_all(question, docs)
    relevant_count = count_relevant(grades)
    
    # If enough relevant docs
    if relevant_count >= threshold:
        break
    
    # Else rewrite query
    failure_reason = analyze_grades(grades)
    query = rewrite_query(query, failure_reason)
    retry_count += 1

# Generate answer from best documents
answer = generate_answer(question, docs)

# Check for hallucinations
if hallucination_detected(answer, docs):
    # Refine with constraints
    answer = refine_answer(answer, docs)

return answer
```

### 9. **Hallucination Detection via Grounding**

**Purpose**: Ensure answer claims are supported by sources

**Algorithm**:
```
for each claim in answer:
    # Find supporting evidence
    supporting_text = search_sources(claim)
    
    if supporting_text found:
        grounding_level = calculate_similarity(claim, supporting_text)
        if grounding_level > threshold:
            mark_as_grounded(claim)
        else:
            mark_as_weakly_grounded(claim)
    else:
        # Not in sources
        mark_as_hallucinated(claim)

# Classify hallucination level
hallucination_level = classify_from_marks()
confidence = 1.0 - hallucination_fraction()
```

### 10. **Recursive Query Decomposition**

**Purpose**: Break complex questions into answerable sub-queries

**Algorithm**:
```
function decompose_query(query):
    # Check complexity
    if is_simple(query):
        return [query]
    
    # Identify question structure
    questions = extract_questions(query)
    
    # Recursively decompose
    all_subqueries = []
    for q in questions:
        subqueries = decompose_query(q)
        all_subqueries.extend(subqueries)
    
    return all_subqueries

# Example
Input:  "Who has liability if payment is late and interest accrues?"
Output: [
    "Who has liability?",
    "What happens if payment is late?",
    "How does interest accrue?"
]
```

---

## System Resilience & Error Handling

### Fallback Mechanisms

| Layer | Primary | Fallback 1 | Fallback 2 |
|-------|---------|-----------|-----------|
| **Retrieval** | Hybrid (semantic + BM25) | BM25 only | Simple lexical match |
| **Grading** | LLM-based | Keyword matching | Default threshold |
| **Embeddings** | Ollama embedder | Precomputed cache | Fallback vectors |
| **LLM Generation** | Llama2 | Summarize without LLM | Extract template answer |

### Error Recovery

```python
def robust_operation(operation, max_retries=3, backoff_factor=2):
    """
    Execute operation with exponential backoff retry
    """
    for attempt in range(max_retries):
        try:
            return operation()
        except TemporaryError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor ** attempt
            logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s")
            time.sleep(wait_time)
        except PermanentError:
            logger.error(f"Permanent error, triggering fallback")
            return fallback_operation()
```

---

## Performance Considerations

### Optimization Strategies

1. **Embedding Caching**
   - Cache query embeddings
   - Cache document embeddings
   - LRU cache with size limits

2. **Search Optimization**
   - Pre-index common queries
   - Use approximate nearest neighbors (HNSW)
   - Batch similarity computations

3. **Chunking Optimization**
   - Process PDFs in parallel
   - Batch embedding generation
   - Streaming chunk creation

4. **LLM Optimization**
   - Use smaller models where possible
   - Batch LLM calls
   - Cache prompt templates
   - Reuse context windows

### Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| PDF ingestion | 30-60s per 100-page doc | Includes vision processing |
| Query expansion | <100ms | Ontology lookup |
| Hybrid retrieval | 500ms | Top-5 retrieval |
| Document grading | 2-5s | 5 docs × 500ms each |
| Answer generation | 3-10s | LLM generation |
| Hallucination check | 2-3s | LLM validation |
| **Total latency** | **8-25s** | Depends on retries |

---

## Security & Privacy

### Data Protection
- **Local Models**: No data leaves the system
- **Private Vector DB**: Chroma runs locally
- **No API Calls**: No external LLM services
- **Encrypted Storage**: Optional encryption for sensitive docs

### Access Control
- Session-based authentication
- Role-based document access (future)
- Audit logging of queries
- PII detection and redaction (future)

### Model Safety
- Temperature tuning (low = deterministic)
- Token limits prevent runaway generation
- Hallucination detection as final check
- Human review workflows (future)

---

## Scalability & Deployment

### Horizontal Scaling

```
Load Balancer
├── API Server 1
├── API Server 2
└── API Server N

Shared Backend:
├── Chroma Vector DB (multi-client support)
├── SQLite Metadata (atomic operations)
└── File Storage (shared mount)

Local Instances:
├── Ollama instance 1
├── Ollama instance 2
└── Ollama instance N
```

### Resource Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB (OS + models)
- Models: 8 GB (llama2 7B)

**Recommended**:
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 50+ GB (OS + models + data)
- Models: 20 GB (llama2 7B + llava 13B)

**High Performance**:
- CPU: 16+ cores
- RAM: 64+ GB
- GPU: CUDA-enabled (optional)
- Disk: 100+ GB SSD
- Models: GPU acceleration

---

## Future Enhancements

### Planned Features
1. **Multi-User Management**
   - User authentication
   - Document access control
   - Query history tracking

2. **Advanced Retrieval**
   - Hierarchical document structure
   - Temporal document versioning
   - Graph-based entity relationships

3. **Enhanced Generation**
   - Multi-language support
   - Summarization modes
   - Custom response formatting

4. **Monitoring & Analytics**
   - Query metrics and trends
   - Performance dashboards
   - Cost tracking
   - User analytics

5. **Advanced Safety**
   - PII detection and redaction
   - Content moderation
   - Bias detection
   - Compliance reporting

---

## Conclusion

The **Multi-Modal Legal AI Chatbot System** represents a sophisticated integration of modern NLP techniques, vector-based retrieval, and state-machine orchestration to deliver accurate, verifiable legal information. Key design principles include:

- **Accuracy First**: Multiple validation layers prevent hallucinations
- **Transparency**: All answers include citations and sources
- **Resilience**: Fallback mechanisms for every critical operation
- **Privacy**: Local models ensure data stays within the organization
- **Scalability**: Modular architecture supports growth
- **Maintainability**: Clear separation of concerns and design patterns

The system is production-ready and can handle complex legal document analysis at enterprise scale.

---

**Document Version**: 1.0  
**Last Updated**: May 2026  
**Status**: Complete
