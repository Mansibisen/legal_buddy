"""
FastAPI Application for Multi-Modal Legal System
Provides REST endpoints for document ingestion, search, and retrieval
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from ingestion.data import MultiModalLegalIngestionPipeline, IngestionStats
from ingestion.vector_store import VectorStore
from retriever.retrieval_integration import LegalRetrievalEngine
from selfCorrection.grader import DocumentGrader
from selfCorrection.rewriter import QueryRewriter, RetryStrategy
from selfCorrection.hallucination_checker import HallucinationChecker
from answer_generator import MultiModalAnswerGenerator, AnswerFormatter
from selfCorrection.self_correction_orchestrator import SelfCorrectionOrchestrator, SelfCorrectionWorkflowState
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal Legal System API",
    description="API for ingesting, processing, and searching legal documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store: Optional[VectorStore] = None
ingestion_pipeline: Optional[MultiModalLegalIngestionPipeline] = None
retrieval_engine: Optional[LegalRetrievalEngine] = None
document_grader: Optional[DocumentGrader] = None
query_rewriter: Optional[QueryRewriter] = None
hallucination_checker: Optional[HallucinationChecker] = None
answer_generator: Optional[MultiModalAnswerGenerator] = None
orchestrator: Optional[SelfCorrectionOrchestrator] = None


# ==================== Request/Response Models ====================

class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    k: int = 5
    filter_section: Optional[str] = None


class SearchResult(BaseModel):
    """Single search result"""
    content: str
    relevance_score: float
    section_title: str
    chunk_type: str
    page: Optional[int] = None
    document_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    timestamp: str


class IngestionRequest(BaseModel):
    """Document ingestion request"""
    pdf_path: Optional[str] = None
    document_id: Optional[str] = None
    skip_images: bool = False


class IngestionResponse(BaseModel):
    """Ingestion response"""
    status: str
    document_id: str
    chunks_created: int
    images_processed: int
    error: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    vector_store_ready: bool
    documents_count: int
    timestamp: str


# ==================== Self-Correction Models ====================

class SelfCorrectionRequest(BaseModel):
    """Self-correction workflow request"""
    question: str
    max_attempts: int = 3
    check_hallucinations: bool = True
    include_images: bool = True


class GradeAssessment(BaseModel):
    """Document grading assessment"""
    grade: str  # RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT
    score: int  # 0-3
    relevant_count: int
    total_count: int
    confidence: float


class RewriteInfo(BaseModel):
    """Query rewrite information"""
    original_query: str
    rewritten_query: str
    strategy: str
    explanation: str
    confidence: float


class HallucinationInfo(BaseModel):
    """Hallucination check information"""
    level: str  # none, minor, moderate, severe
    is_hallucinating: bool
    confidence: float
    hallucinated_claims: List[str] = []
    recommendations: List[str] = []


class WorkflowMetrics(BaseModel):
    """Workflow execution metrics"""
    total_attempts: int
    retrieval_time: float
    grading_time: float
    rewriting_time: float
    generation_time: float
    total_time: float


class SelfCorrectionResponse(BaseModel):
    """Self-correction workflow response"""
    question: str
    answer: str
    success: bool
    attempts: int
    grade_assessment: Optional[GradeAssessment] = None
    rewrites: List[RewriteInfo] = []
    hallucination_assessment: Optional[HallucinationInfo] = None
    sources: List[str] = []
    citations: List[Dict[str, Any]] = []
    metrics: Optional[WorkflowMetrics] = None
    timestamp: str = ""


class ChatMessageRequest(BaseModel):
    """Chat message request"""
    question: str
    session_id: str = "default"
    max_attempts: int = 3
    check_hallucinations: bool = True



# ==================== Initialization ====================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_store, ingestion_pipeline, retrieval_engine
    global document_grader, query_rewriter, hallucination_checker
    global answer_generator, orchestrator
    
    logger.info("Initializing application...")
    
    try:
        vector_store = VectorStore()
        ingestion_pipeline = MultiModalLegalIngestionPipeline()
        retrieval_engine = LegalRetrievalEngine()
        
        # Initialize self-correction components
        document_grader = DocumentGrader()
        query_rewriter = QueryRewriter()
        hallucination_checker = HallucinationChecker()
        answer_generator = MultiModalAnswerGenerator()
        orchestrator = SelfCorrectionOrchestrator()
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")


# ==================== Health & Status Endpoints ====================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check application health and vector store status"""
    try:
        info = vector_store.get_collection_info()
        
        return HealthCheck(
            status="healthy",
            vector_store_ready=True,
            documents_count=info.get("count", 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            vector_store_ready=False,
            documents_count=0,
            timestamp=datetime.now().isoformat()
        )


@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        info = vector_store.get_collection_info()
        
        return {
            "status": "ready",
            "vector_store": {
                "collection": info.get("name"),
                "documents": info.get("count", 0),
                "path": settings.CHROMA_DB_PATH
            },
            "config": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "vision_enabled": settings.VISION_ENABLED,
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Search Endpoints ====================

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery):
    """
    Search for documents in vector store
    
    Args:
        search_query: Search query with optional filters
        
    Returns:
        List of relevant documents with scores
    """
    try:
        logger.info(f"Searching for: {search_query.query}")
        
        results = vector_store.search(
            query=search_query.query,
            k=search_query.k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                content=result["content"],
                relevance_score=result["relevance_score"],
                section_title=result["metadata"].get("section_title", "N/A"),
                chunk_type=result["metadata"].get("chunk_type", "text"),
                page=result["metadata"].get("page"),
                document_id=result["metadata"].get("document_id")
            ))
        
        return SearchResponse(
            query=search_query.query,
            results=formatted_results,
            total_results=len(formatted_results),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/legal")
async def legal_search(query: str, k: int = 5):
    """
    Perform legal-specific search
    Optimized prompts for common legal queries
    """
    # Map common legal searches
    legal_queries_map = {
        "liability": "limitation of liability indemnification",
        "confidentiality": "confidential information non-disclosure",
        "termination": "termination clause cancellation",
        "ip": "intellectual property copyright patent trademark",
        "payment": "payment terms price cost consideration",
        "jurisdiction": "governing law jurisdiction venue",
    }
    
    # Expand query if it matches a legal category
    expanded_query = legal_queries_map.get(query.lower(), query)
    
    search_query = SearchQuery(query=expanded_query, k=k)
    return await search_documents(search_query)


@app.get("/search/suggest")
async def search_suggestions():
    """Get suggested legal search topics"""
    suggestions = [
        "limitation of liability",
        "confidentiality clauses",
        "termination provisions",
        "intellectual property rights",
        "payment terms",
        "governing law and jurisdiction",
        "indemnification",
        "warranties and representations",
        "force majeure",
        "notices and communications"
    ]
    
    return {
        "suggestions": suggestions,
        "description": "Common legal document searches"
    }


# ==================== Advanced Retrieval Endpoints ====================

@app.post("/retrieve/question")
async def answer_legal_question(question: str, k: int = 5, verbose: bool = False):
    """
    Answer a legal question using hybrid retrieval
    
    Args:
        question: Legal question
        k: Number of results
        verbose: Include query expansion details
        
    Returns:
        Answer with supporting documents
    """
    try:
        logger.info(f"Answering question: {question}")
        
        result = retrieval_engine.answer_legal_question(question, k=k, verbose=verbose)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/hybrid")
async def hybrid_retrieve(
    query: str,
    k: int = 5,
    expand_query: bool = True,
    method: str = "hybrid"
):
    """
    Retrieve documents using hybrid search
    Combines semantic similarity with keyword matching
    
    Args:
        query: Search query
        k: Number of results
        expand_query: Expand query with legal terms
        method: "semantic", "keyword", or "hybrid"
        
    Returns:
        List of retrieved documents with scores
    """
    try:
        logger.info(f"Hybrid retrieve: {query} (method: {method})")
        
        results = retrieval_engine.search_with_method(
            query=query,
            method=method,
            k=k,
            expand_query=expand_query
        )
        
        return {
            "query": query,
            "method": method,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Hybrid retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/category")
async def retrieve_by_category(category: str, k: int = 5):
    """
    Retrieve documents by legal category
    
    Available categories:
    - liability
    - payment
    - confidentiality
    - termination
    - ip (intellectual property)
    - warranties
    - jurisdiction
    - obligations
    
    Args:
        category: Legal category
        k: Number of results
        
    Returns:
        Category-specific documents
    """
    try:
        logger.info(f"Category retrieve: {category}")
        
        result = retrieval_engine.search_by_category(category, k=k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/compare")
async def compare_clauses(
    clause1_query: str,
    clause2_query: str,
    k: int = 3
):
    """
    Compare two different clauses or concepts
    
    Args:
        clause1_query: Query for first clause
        clause2_query: Query for second clause
        k: Number of results per clause
        
    Returns:
        Side-by-side comparison
    """
    try:
        logger.info(f"Comparing clauses")
        
        result = retrieval_engine.compare_clauses(clause1_query, clause2_query, k=k)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Compare clauses error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/related")
async def find_related_clauses(clause_content: str, k: int = 5):
    """
    Find clauses related to a given clause
    
    Args:
        clause_content: Content of the reference clause
        k: Number of related clauses
        
    Returns:
        List of related clauses
    """
    try:
        logger.info(f"Finding related clauses")
        
        results = retrieval_engine.find_related_clauses(clause_content, k=k)
        
        return {
            "success": True,
            "related_clauses": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Find related clauses error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve/advanced")
async def advanced_retrieve(
    query: str,
    k: int = 5,
    section_title: Optional[str] = None,
    chunk_type: Optional[str] = None,
    document_id: Optional[str] = None,
    relevance_threshold: float = 0.3
):
    """
    Advanced search with multiple filters
    
    Args:
        query: Search query
        k: Number of results
        section_title: Optional section filter
        chunk_type: Optional chunk type filter (text, image_description, table)
        document_id: Optional document filter
        relevance_threshold: Minimum relevance score (0-1)
        
    Returns:
        Filtered search results
    """
    try:
        logger.info(f"Advanced retrieve: {query}")
        
        results = retrieval_engine.advanced_search(
            query=query,
            k=k,
            section_title=section_title,
            chunk_type=chunk_type,
            document_id=document_id,
            relevance_threshold=relevance_threshold
        )
        
        return {
            "success": True,
            "query": query,
            "filters": {
                "section_title": section_title,
                "chunk_type": chunk_type,
                "document_id": document_id,
                "relevance_threshold": relevance_threshold
            },
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve/suggestions")
async def get_query_suggestions(partial_query: str):
    """
    Get suggestions for completing a query
    
    Args:
        partial_query: Partial search query
        
    Returns:
        List of suggested completions
    """
    try:
        logger.info(f"Getting suggestions for: {partial_query}")
        
        suggestions = retrieval_engine.get_search_suggestions(partial_query)
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Ingestion Endpoints ====================

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    Ingest a document for processing
    
    Args:
        request: Ingestion request with PDF path
        
    Returns:
        Ingestion status and results
    """
    if not request.pdf_path:
        raise HTTPException(status_code=400, detail="pdf_path is required")
    
    try:
        logger.info(f"Ingesting document: {request.pdf_path}")
        
        result = ingestion_pipeline.ingest_pdf_file(
            pdf_path=request.pdf_path,
            document_id=request.document_id,
            skip_images=request.skip_images
        )
        
        if result["status"] == "success":
            return IngestionResponse(
                status="success",
                document_id=result["document_id"],
                chunks_created=result["chunks_created"],
                images_processed=result["images_processed"]
            )
        else:
            return IngestionResponse(
                status="error",
                document_id="",
                chunks_created=0,
                images_processed=0,
                error=result.get("error", "Unknown error")
            )
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/batch")
async def ingest_batch(directory: str, max_files: Optional[int] = None, skip_images: bool = False):
    """
    Ingest multiple documents from a directory
    
    Args:
        directory: Directory containing PDF files
        max_files: Maximum number of files to process
        skip_images: Skip image processing
        
    Returns:
        Batch ingestion summary
    """
    try:
        logger.info(f"Starting batch ingestion from: {directory}")
        
        results = ingestion_pipeline.ingest_pdf_directory(
            directory=directory,
            skip_images=skip_images,
            max_files=max_files
        )
        
        summary = ingestion_pipeline.get_ingestion_summary()
        summary["results"] = results
        
        return summary
        
    except Exception as e:
        logger.error(f"Batch ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def upload_and_ingest(file: UploadFile = File(...), document_id: Optional[str] = None):
    """
    Upload and ingest a PDF file
    
    Args:
        file: PDF file to upload
        document_id: Optional document ID
        
    Returns:
        Ingestion result
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Ingest
        result = ingestion_pipeline.ingest_pdf_file(
            pdf_path=temp_path,
            document_id=document_id or file.filename.split(".")[0]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Upload ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Collection Management ====================

@app.delete("/collection/clear")
async def clear_collection():
    """Clear all documents from collection (use with caution)"""
    try:
        success = vector_store.clear_collection()
        
        return {
            "status": "success" if success else "error",
            "message": "Collection cleared" if success else "Failed to clear collection"
        }
        
    except Exception as e:
        logger.error(f"Clear collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/stats")
async def get_collection_stats():
    """Get collection statistics"""
    try:
        info = vector_store.get_collection_info()
        
        return {
            "name": info.get("name"),
            "document_count": info.get("count", 0),
            "embedding_model": settings.EMBEDDING_MODEL,
            "path": settings.CHROMA_DB_PATH,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat Endpoints ====================

@app.post("/chat/message")
async def chat_message(request: ChatMessageRequest):
    """
    Main chat endpoint for legal questions
    Orchestrates full self-correction workflow
    
    Args:
        request: ChatMessageRequest with question and options
    """
    try:
        logger.info(f"Chat message from {request.session_id}: {request.question}")
        
        # Define wrapper functions for orchestrator
        def retriever_fn(query):
            try:
                logger.info(f"Retriever function called with query: {query}")
                results = retrieval_engine.search_with_method(
                    query,
                    method="hybrid",
                    k=5
                )
                if not results:
                    logger.warning(f"No results from retrieval engine for query: {query}")
                    return []
                
                docs = [r.get("content", "") for r in results] if isinstance(results, list) else []
                logger.info(f"Retrieved {len(docs)} documents. First doc length: {len(docs[0]) if docs else 0} chars")
                return docs
            except Exception as e:
                logger.error(f"Retrieval error: {e}", exc_info=True)
                return []
        
        def grader_fn(q, docs):
            try:
                logger.info(f"Grader function called with {len(docs)} documents")
                result = document_grader.grade_documents(q, docs)
                
                # Count RELEVANT and PARTIAL as relevant
                relevant = sum(1 for r in result if r.get("grade", "").upper() in ["RELEVANT", "PARTIAL"])
                logger.info(f"Detailed grades: {[r.get('grade') for r in result]}")
                logger.info(f"Grading complete: {relevant} relevant out of {len(docs)}")
                
                # Extract relevant documents (keep original document text, not just preview)
                relevant_docs = []
                for i, grading_result in enumerate(result):
                    if grading_result.get("grade", "").upper() in ["RELEVANT", "PARTIAL"]:
                        # Use original full document, not just the preview
                        relevant_docs.append(docs[i])
                
                return {
                    "relevant_count": relevant,
                    "total_documents": len(docs),
                    "relevant_documents": relevant_docs,
                    "feedback": f"Found {relevant} relevant documents out of {len(docs)}"
                }
            except Exception as e:
                logger.error(f"Grading error: {e}", exc_info=True)
                return {"relevant_count": 0, "total_documents": len(docs), "relevant_documents": []}
        
        def rewriter_fn(orig_query, reason):
            try:
                logger.info(f"Rewriter function called. Original query: {orig_query}, Reason: {reason}")
                result = query_rewriter.rewrite_query(orig_query, reason)
                logger.info(f"Query rewritten: {result.get('rewritten_query', orig_query)}")
                return result
            except Exception as e:
                logger.error(f"Rewriting error: {e}", exc_info=True)
                return {"rewritten_query": orig_query}
        
        def generator_fn(q, docs):
            try:
                logger.info(f"Generator function called with question: {q}, doc_count: {len(docs)}")
                result = answer_generator.generate_answer(q, docs)
                logger.info(f"Answer generated: {result.answer_text[:100] if result.answer_text else 'None'}")
                return result.answer_text
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                return "Unable to generate answer"
        
        def hallucination_fn(q, ans, docs):
            try:
                logger.info(f"Hallucination checker called for answer length: {len(ans) if ans else 0}")
                result = hallucination_checker.check_hallucination(q, ans, docs)
                logger.info(f"Hallucination check complete: {result}")
                return result
            except Exception as e:
                logger.error(f"Hallucination check error: {e}", exc_info=True)
                return {"is_hallucinating": False, "hallucination_level": "unknown"}
        
        # Run workflow
        workflow_state = orchestrator.execute_workflow(
            request.question,
            retriever_fn,
            grader_fn,
            rewriter_fn,
            generator_fn,
            hallucination_fn
        )
        
        logger.info(f"Workflow completed. Success: {workflow_state.success}, Answer: {workflow_state.answer[:50] if workflow_state.answer else 'None'}")
        
        # Ensure all values are JSON-serializable
        relevant_docs = workflow_state.relevant_documents or []
        sources = [str(doc) for doc in relevant_docs[:3]] if relevant_docs else []
        
        response = {
            "success": workflow_state.success,
            "answer": workflow_state.answer or "Unable to generate answer",
            "citations": [
                {
                    "source_id": f"Doc {i+1}",
                    "page": None,
                    "confidence": 0.8
                }
                for i in range(min(5, len(relevant_docs)))
            ],
            "sources": sources,
            "metrics": {
                "total_attempts": int(workflow_state.attempt_count),
                "retrieval_time": float(workflow_state.retrieval_time),
                "grading_time": float(workflow_state.grading_time),
                "rewriting_time": float(workflow_state.rewriting_time),
                "generation_time": float(workflow_state.generation_time),
                "total_time": float(workflow_state.total_time)
            },
            "session_id": request.session_id
        }
        
        logger.info(f"Chat response prepared successfully")
        return response
        
    except Exception as e:
        import traceback
        logger.error(f"Chat error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "answer": "An error occurred while processing your question.",
            "session_id": request.session_id
        }


@app.get("/chat/history")
async def chat_history(session_id: str = "default"):
    """Get chat history for a session"""
    try:
        return {
            "session_id": session_id,
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/session")
async def create_session():
    """Create a new chat session"""
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        return {
            "session_id": session_id,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chatbot")
async def chatbot_ui():
    """Serve chatbot UI"""
    from fastapi.responses import FileResponse
    return FileResponse("templates/chatbot.html")


# ==================== Self-Correction Endpoints ====================

@app.post("/self-correct")
async def self_correct_workflow(request: SelfCorrectionRequest) -> SelfCorrectionResponse:
    """
    Execute complete self-correction workflow
    
    Orchestrates: retrieve -> grade -> (rewrite if needed) -> generate -> hallucination check
    
    Args:
        request: SelfCorrectionRequest with question and options
        
    Returns:
        SelfCorrectionResponse with answer, grading, rewrites, and metrics
    """
    try:
        logger.info(f"Starting self-correction workflow for: {request.question}")
        
        # Define helper functions for orchestrator
        def retrieve_fn(query: str):
            try:
                return retrieval_engine.answer_legal_question(
                    question=query,
                    top_k=5
                ).get("source_documents", [])
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                return []
        
        def grade_fn(question: str, documents: List[str]):
            try:
                results = document_grader.grade_documents(question, documents)
                relevant = [r for r in results if r.get("grade") == "RELEVANT"]
                return {
                    "grade_assessment": results,
                    "relevant_documents": [r.get("document") for r in relevant],
                    "relevant_count": len(relevant),
                    "total_documents": len(documents),
                    "feedback": f"Found {len(relevant)} relevant documents out of {len(documents)}"
                }
            except Exception as e:
                logger.error(f"Grading error: {e}")
                return {"relevant_count": 0, "total_documents": len(documents), "feedback": str(e)}
        
        def rewrite_fn(query: str, reason: str):
            try:
                return query_rewriter.rewrite_query(query, reason)
            except Exception as e:
                logger.error(f"Rewrite error: {e}")
                return {"rewritten_query": query, "explanation": str(e)}
        
        def generate_fn(question: str, documents: List[str]):
            try:
                result = answer_generator.generate_answer(question, documents)
                return result.answer_text
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error generating answer: {str(e)}"
        
        def hallucination_fn(question: str, answer: str, documents: List[str]):
            try:
                return hallucination_checker.check_hallucination(
                    question, answer, documents
                )
            except Exception as e:
                logger.error(f"Hallucination check error: {e}")
                return {
                    "hallucination_level": "unknown",
                    "is_hallucinating": False,
                    "confidence": 0.0
                }
        
        # Execute workflow
        workflow_state = orchestrator.execute_workflow(
            question=request.question,
            retriever_fn=retrieve_fn,
            grader_fn=grade_fn,
            rewriter_fn=rewrite_fn,
            generator_fn=generate_fn,
            hallucination_checker_fn=hallucination_fn
        )
        
        # Build response
        rewrites = [
            RewriteInfo(
                original_query=r.get("original_query", ""),
                rewritten_query=r.get("rewritten_query", ""),
                strategy=r.get("strategy_used", ""),
                explanation=r.get("explanation", ""),
                confidence=r.get("confidence", 0.0)
            )
            for r in workflow_state.rewrite_history
        ]
        
        grade_info = None
        if workflow_state.grade_assessment:
            grade_info = GradeAssessment(
                grade=workflow_state.grade_assessment.get("grade", "UNKNOWN"),
                score=workflow_state.grade_assessment.get("score", 0),
                relevant_count=workflow_state.relevant_count,
                total_count=len(workflow_state.retrieved_documents),
                confidence=workflow_state.grade_assessment.get("confidence", 0.0)
            )
        
        hallucination_info = None
        if workflow_state.hallucination_assessment:
            hallucination_info = HallucinationInfo(
                level=workflow_state.hallucination_assessment.get("hallucination_level", "unknown"),
                is_hallucinating=workflow_state.hallucination_assessment.get("is_hallucinating", False),
                confidence=workflow_state.hallucination_assessment.get("confidence", 0.0),
                hallucinated_claims=workflow_state.hallucination_assessment.get("hallucinated_claims", []),
                recommendations=workflow_state.hallucination_assessment.get("recommendations", [])
            )
        
        metrics = WorkflowMetrics(
            total_attempts=workflow_state.attempt_count,
            retrieval_time=workflow_state.retrieval_time,
            grading_time=workflow_state.grading_time,
            rewriting_time=workflow_state.rewriting_time,
            generation_time=workflow_state.generation_time,
            total_time=workflow_state.total_time
        )
        
        return SelfCorrectionResponse(
            question=request.question,
            answer=workflow_state.answer or "",
            success=workflow_state.success,
            attempts=workflow_state.attempt_count,
            grade_assessment=grade_info,
            rewrites=rewrites,
            hallucination_assessment=hallucination_info,
            sources=workflow_state.relevant_documents,
            citations=[],  # Would be populated from answer_generator
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Self-correction workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/self-correct/status/{request_id}")
async def get_workflow_status(request_id: str):
    """Get status of a self-correction workflow (if tracking implemented)"""
    return {
        "request_id": request_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/grade/documents")
async def grade_documents(question: str, documents: List[str]):
    """
    Grade retrieved documents for relevance
    
    Args:
        question: The question being asked
        documents: List of document chunks to grade
        
    Returns:
        Grade assessment for each document
    """
    try:
        results = document_grader.grade_documents(question, documents)
        return {
            "question": question,
            "gradings": results,
            "summary": {
                "relevant": len([r for r in results if r.get("grade") == "RELEVANT"]),
                "partial": len([r for r in results if r.get("grade") == "PARTIALLY_RELEVANT"]),
                "irrelevant": len([r for r in results if r.get("grade") == "IRRELEVANT"])
            }
        }
    except Exception as e:
        logger.error(f"Grading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewrite/query")
async def rewrite_query_endpoint(query: str, reason: str, strategy: str = "expand"):
    """
    Rewrite a query that failed to retrieve relevant documents
    
    Args:
        query: Original query
        reason: Why it failed
        strategy: Rewriting strategy
        
    Returns:
        Rewritten query with explanation
    """
    try:
        result = query_rewriter.rewrite_query(query, reason, strategy=strategy)
        return result
    except Exception as e:
        logger.error(f"Rewrite error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check/hallucinations")
async def check_hallucinations(question: str, answer: str, documents: List[str]):
    """
    Check if answer contains hallucinations
    
    Args:
        question: Original question
        answer: Generated answer
        documents: Source documents
        
    Returns:
        Hallucination assessment
    """
    try:
        result = hallucination_checker.check_hallucination(question, answer, documents)
        return result
    except Exception as e:
        logger.error(f"Hallucination check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/answer")
async def generate_answer_endpoint(question: str, documents: List[str]):
    """
    Generate comprehensive answer with citations
    
    Args:
        question: Question to answer
        documents: Source documents
        
    Returns:
        MultiModalAnswer with citations
    """
    try:
        multi_modal_answer = answer_generator.generate_answer(question, documents)
        return multi_modal_answer.to_dict()
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Information Endpoints ====================

@app.get("/info/models")
async def get_model_info():
    """Get information about configured models"""
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "vision_model": settings.VISION_MODEL if settings.VISION_ENABLED else None,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "section_aware_chunking": settings.SECTION_AWARE_CHUNKING,
        "pdf_parser": settings.PDF_EXTRACTION_METHOD,
    }


@app.get("/info/dataset")
async def get_dataset_info():
    """Get information about dataset paths"""
    return {
        "dataset_name": settings.DATASET_NAME,
        "raw_data_dir": settings.RAW_DATA_DIR,
        "processed_data_dir": settings.PROCESSED_DATA_DIR,
        "cache_dir": settings.DATASET_CACHE_DIR,
    }


# ==================== Root Endpoint ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Modal Legal System API",
        "version": "1.0.0",
        "description": "API for processing and searching legal documents",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "search": {
                "basic": "/search",
                "legal": "/search/legal",
                "suggestions": "/search/suggest"
            },
            "ingestion": {
                "single": "/ingest",
                "batch": "/ingest/batch",
                "upload": "/ingest/upload"
            },
            "collection": {
                "stats": "/collection/stats",
                "clear": "/collection/clear"
            },
            "retrieval": {
                "question": "/retrieve/question",
                "hybrid": "/retrieve/hybrid",
                "category": "/retrieve/category",
                "compare": "/retrieve/compare",
                "related": "/retrieve/related",
                "advanced": "/retrieve/advanced"
            },
            "self_correction": {
                "workflow": "/self-correct",
                "grade": "/grade/documents",
                "rewrite": "/rewrite/query",
                "hallucination_check": "/check/hallucinations",
                "generate": "/generate/answer",
                "status": "/self-correct/status/{request_id}"
            },
            "info": {
                "models": "/info/models",
                "dataset": "/info/dataset"
            }
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
