#!/usr/bin/env python3
"""
Standalone document ingestion script
Run this independently from the chat application

Usage:
    python ingest_documents.py
    
This will ingest all PDF files from ./data/raw/ directory
"""
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.data import MultiModalLegalIngestionPipeline
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print formatted header"""
    width = 60
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def print_success(text: str):
    """Print success message"""
    print(f"\n✅ {text}")


def print_error(text: str):
    """Print error message"""
    print(f"\n❌ {text}")


def print_info(text: str):
    """Print info message"""
    print(f"\n📝 {text}")


def ingest_from_directory():
    """Ingest all PDFs from ./data/raw directory"""
    print_header("DOCUMENT INGESTION - DIRECTORY MODE")
    
    # Check if directory exists
    data_dir = Path("./data/raw")
    if not data_dir.exists():
        print_error("Directory ./data/raw not found!")
        print("   Please create it and add PDF files:")
        print("   mkdir -p ./data/raw")
        print("   cp your-documents/*.pdf ./data/raw/")
        return False
    
    # Check if there are any PDFs
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print_error("No PDF files found in ./data/raw/")
        return False
    
    print_info(f"Found {len(pdf_files)} PDF file(s) to ingest:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    
    # Initialize pipeline
    print_info("Initializing ingestion pipeline...")
    pipeline = MultiModalLegalIngestionPipeline()
    
    # Ingest all documents
    print_info("Starting batch ingestion...")
    print("-" * 60)
    
    try:
        results = pipeline.ingest_pdf_directory(
            directory="./data/raw",
            skip_images=False,  # Set to True to speed up (skip image processing)
            max_files=None  # Set to number to limit files
        )
        
        print("-" * 60)
        
        # Get summary
        summary = pipeline.get_ingestion_summary()
        
        print_success("Ingestion complete!")
        print("\n📊 INGESTION SUMMARY:")
        print("-" * 60)
        print(f"   • Documents processed: {summary.get('total_documents', 0)}")
        print(f"   • Total chunks created: {summary.get('total_chunks', 0)}")
        print(f"   • Total images processed: {summary.get('total_images_processed', 0)}")
        print(f"   • Processing time: {summary.get('total_processing_time', 0):.2f}s")
        print(f"   • Database path: {settings.CHROMA_DB_PATH}")
        print("-" * 60)
        
        # Show next steps
        print("\n📌 NEXT STEPS:")
        print("   1. Keep the chat app running: python app.py")
        print("   2. Open browser: http://localhost:8000/chatbot")
        print("   3. Start asking questions about your documents!")
        
        return True
        
    except Exception as e:
        print_error(f"Ingestion failed: {str(e)}")
        logger.exception("Ingestion error:")
        return False


def ingest_single_file(file_path: str):
    """Ingest a single PDF file"""
    print_header("DOCUMENT INGESTION - SINGLE FILE MODE")
    
    # Check if file exists
    pdf_file = Path(file_path)
    if not pdf_file.exists():
        print_error(f"File not found: {file_path}")
        return False
    
    if not pdf_file.suffix.lower() == ".pdf":
        print_error(f"File is not a PDF: {file_path}")
        return False
    
    print_info(f"Ingesting: {pdf_file.name}")
    
    # Initialize pipeline
    print_info("Initializing ingestion pipeline...")
    pipeline = MultiModalLegalIngestionPipeline()
    
    # Ingest the document
    print_info("Starting ingestion...")
    print("-" * 60)
    
    try:
        result = pipeline.ingest_pdf_file(
            pdf_path=str(pdf_file),
            skip_images=False
        )
        
        print("-" * 60)
        
        if result.get("status") == "success":
            print_success(f"Successfully ingested: {pdf_file.name}")
            print("\n📊 INGESTION RESULTS:")
            print("-" * 60)
            print(f"   • Chunks created: {result.get('chunks_created', 0)}")
            print(f"   • Images processed: {result.get('images_processed', 0)}")
            print(f"   • Processing time: {result.get('processing_time', 0):.2f}s")
            print("-" * 60)
            return True
        else:
            print_error(f"Ingestion failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print_error(f"Ingestion failed: {str(e)}")
        logger.exception("Ingestion error:")
        return False


def main():
    """Main function"""
    print_header("MULTI-MODAL LEGAL SYSTEM - DOCUMENT INGESTION")
    
    print("\n📚 INGESTION MODES:")
    print("   1. Directory mode  (ingest all PDFs from ./data/raw/)")
    print("   2. Single file mode (ingest one specific PDF)")
    
    try:
        choice = input("\n🔹 Choose mode (1 or 2, default 1): ").strip() or "1"
        
        if choice == "1":
            success = ingest_from_directory()
        elif choice == "2":
            file_path = input("🔹 Enter PDF file path: ").strip()
            success = ingest_single_file(file_path)
        else:
            print_error("Invalid choice. Please enter 1 or 2.")
            success = False
        
        if success:
            print("\n" + "=" * 60)
            print("✅ READY TO USE YOUR DOCUMENTS!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("❌ INGESTION FAILED - Please check the errors above")
            print("=" * 60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_error("Ingestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
