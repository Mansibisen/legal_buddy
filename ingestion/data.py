"""
Main Data Ingestion Pipeline for Multi-Modal Legal System
Orchestrates the complete process:
1. Load dataset (Pile-of-Law)
2. Parse PDFs (with images)
3. Clean text
4. Process images with vision model
5. Chunk intelligently
6. Create embeddings and store in vector DB
"""
import logging
import logging.handlers
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from tqdm import tqdm

from config import settings
from pdf_parser import parse_pdf, ImageExtractor
from vision_processor import process_document_images
from text_cleaner import clean_document
from chunking import chunk_document, HybridChunker
from vector_store import VectorStore

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging"""
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger_obj = logging.getLogger()
    logger_obj.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger_obj.addHandler(console_handler)
    logger_obj.addHandler(file_handler)


@dataclass
class IngestionStats:
    """Track ingestion statistics"""
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    images_processed: int = 0
    total_tokens: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "chunks_created": self.chunks_created,
            "images_processed": self.images_processed,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds(),
            "avg_chunks_per_document": (
                self.chunks_created / self.documents_processed
                if self.documents_processed > 0 else 0
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class MultiModalLegalIngestionPipeline:
    """
    Main pipeline for ingesting multi-modal legal documents
    """
    
    def __init__(self, use_async_vision: bool = False):
        """
        Initialize pipeline
        
        Args:
            use_async_vision: Use async vision processing
        """
        self.use_async_vision = use_async_vision
        self.vector_store = VectorStore()
        self.chunker = HybridChunker()
        self.stats = IngestionStats()
        self.stats.start_time = datetime.now()
        
        logger.info("Initialized Multi-Modal Legal Ingestion Pipeline")
    
    def ingest_pdf_file(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
        skip_images: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            document_id: Optional document ID
            skip_images: Skip image processing
            
        Returns:
            Ingestion result with metadata
        """
        logger.info(f"Starting ingestion of: {pdf_path}")
        
        try:
            # Step 1: Parse PDF
            logger.info("Step 1/5: Parsing PDF")
            document = parse_pdf(pdf_path, extract_images=not skip_images)
            
            # Step 2: Clean text
            logger.info("Step 2/5: Cleaning text")
            document = clean_document(document)
            
            # Step 3: Process images
            if not skip_images and document.get("images"):
                logger.info(f"Step 3/5: Processing {len(document['images'])} images")
                document["images"] = process_document_images(
                    document["images"],
                    document_title=pdf_path,
                    use_async=self.use_async_vision
                )
                self.stats.images_processed += len(document["images"])
            else:
                logger.info("Step 3/5: Skipping image processing")
            
            # Step 4: Chunk document
            logger.info("Step 4/5: Chunking document")
            chunks = chunk_document(document, self.chunker)
            self.stats.chunks_created += len(chunks)
            
            # Step 5: Store in vector database
            logger.info("Step 5/5: Creating embeddings and storing")
            doc_id = document_id or Path(pdf_path).stem
            chunk_ids = self.vector_store.add_chunks(chunks, document_id=doc_id)
            
            self.stats.documents_processed += 1
            
            result = {
                "status": "success",
                "document_id": doc_id,
                "pdf_path": pdf_path,
                "chunks_created": len(chunks),
                "images_processed": len(document.get("images", [])),
                "chunk_ids": chunk_ids,
                "metadata": document.get("metadata", {})
            }
            
            logger.info(f"Successfully ingested {pdf_path}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting {pdf_path}: {e}", exc_info=True)
            self.stats.documents_failed += 1
            
            return {
                "status": "error",
                "pdf_path": pdf_path,
                "error": str(e)
            }
    
    def ingest_pdf_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        skip_images: bool = False,
        max_files: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest all PDFs from a directory
        
        Args:
            directory: Directory containing PDFs
            pattern: File pattern (default: *.pdf)
            skip_images: Skip image processing
            max_files: Maximum number of files to process
            
        Returns:
            List of ingestion results
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all matching files
        pdf_files = list(directory_path.glob(pattern))
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to ingest")
        
        results = []
        
        # Process with progress bar
        for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
            result = self.ingest_pdf_file(
                str(pdf_path),
                skip_images=skip_images
            )
            results.append(result)
        
        return results
    
    def ingest_pile_of_law(
        self,
        dataset_path: Optional[str] = None,
        max_files: Optional[int] = None,
        skip_images: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Ingest documents from Pile-of-Law dataset
        
        Args:
            dataset_path: Path to dataset (uses config if not provided)
            max_files: Maximum number of files to process
            skip_images: Skip image processing
            
        Returns:
            List of ingestion results
        """
        dataset_path = dataset_path or settings.RAW_DATA_DIR
        
        logger.info(f"Ingesting Pile-of-Law dataset from: {dataset_path}")
        
        # Ingest all PDFs in dataset
        results = self.ingest_pdf_directory(
            dataset_path,
            pattern="*.pdf",
            skip_images=skip_images,
            max_files=max_files
        )
        
        return results
    
    def get_ingestion_summary(self) -> Dict[str, Any]:
        """Get summary of ingestion process"""
        self.stats.end_time = datetime.now()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats.to_dict(),
            "vector_store_info": self.vector_store.get_collection_info(),
        }
        
        return summary
    
    def save_ingestion_report(self, output_path: str = None):
        """
        Save ingestion report to file
        
        Args:
            output_path: Path to save report (uses config if not provided)
        """
        output_path = output_path or Path(settings.PROCESSED_DATA_DIR) / "ingestion_report.json"
        
        report = self.get_ingestion_summary()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Ingestion report saved to {output_path}")


def run_ingestion_pipeline(
    pdf_source: str,
    mode: str = "directory",
    max_files: Optional[int] = None,
    skip_images: bool = False,
    use_async_vision: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for data ingestion
    
    Args:
        pdf_source: Path to PDF file or directory
        mode: "file" or "directory" or "dataset"
        max_files: Maximum number of files to process
        skip_images: Skip image processing
        use_async_vision: Use async vision processing
        
    Returns:
        Ingestion summary
    """
    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("Starting Multi-Modal Legal Document Ingestion")
    logger.info("=" * 80)
    
    # Initialize pipeline
    pipeline = MultiModalLegalIngestionPipeline(use_async_vision=use_async_vision)
    
    # Run ingestion based on mode
    if mode == "file":
        result = pipeline.ingest_pdf_file(pdf_source)
        results = [result]
        
    elif mode == "directory":
        results = pipeline.ingest_pdf_directory(
            pdf_source,
            skip_images=skip_images,
            max_files=max_files
        )
        
    elif mode == "dataset":
        results = pipeline.ingest_pile_of_law(
            pdf_source,
            max_files=max_files,
            skip_images=skip_images
        )
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Generate summary
    summary = pipeline.get_ingestion_summary()
    summary["results"] = results
    
    # Save report
    pipeline.save_ingestion_report()
    
    # Log summary
    stats = summary["statistics"]
    logger.info("=" * 80)
    logger.info("Ingestion Summary:")
    logger.info(f"  Documents processed: {stats['documents_processed']}")
    logger.info(f"  Documents failed: {stats['documents_failed']}")
    logger.info(f"  Chunks created: {stats['chunks_created']}")
    logger.info(f"  Images processed: {stats['images_processed']}")
    logger.info(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    logger.info(f"  Avg chunks/doc: {stats['avg_chunks_per_document']:.2f}")
    logger.info("=" * 80)
    
    return summary
