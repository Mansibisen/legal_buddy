"""
Multi-Modal PDF Parser for Legal Documents
Handles PDFs with complex layouts, images, and charts
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import pdf2image
from pypdf import PdfReader
import pytesseract

from config import settings

logger = logging.getLogger(__name__)

class PDFParser(ABC):
    """Abstract base class for PDF parsers"""
    
    @abstractmethod
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF and return structured content"""
        pass

class MarkerPDFParser(PDFParser):
    """
    Uses Marker PDF for high-quality PDF parsing
    Preserves markdown structure, handles images and tables
    """
    
    def __init__(self):
        self.name = "marker"
        self.max_pages = settings.MAX_PAGES
        
class MarkerPDFParser(PDFParser):
    """
    Uses PyPDF for PDF text extraction
    """
    
    def __init__(self):
        self.name = "marker"
        self.max_pages = settings.MAX_PAGES
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF using PyPDF
        """
        try:
            logger.info(f"Parsing PDF with PyPDF: {pdf_path}")
            
            pdf_reader = PdfReader(pdf_path)
            text_content = ""
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages[:self.max_pages]):
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()
            
            return {
                "text_content": text_content,
                "pages": len(pdf_reader.pages),
                "metadata": {
                    "source": pdf_path,
                    "parser": self.name,
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF with PyPDF: {e}")
            raise


class UnstructuredPDFParser(PDFParser):
    """
    Alternative PDF parser using PyPDF
    """
    
    def __init__(self):
        self.name = "unstructured"
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using PyPDF"""
        try:
            logger.info(f"Parsing PDF with PyPDF: {pdf_path}")
            
            pdf_reader = PdfReader(pdf_path)
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()
            
            return {
                "text_content": text_content,
                "pages": len(pdf_reader.pages),
                "metadata": {"source": pdf_path, "parser": self.name}
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF with PyPDF: {e}")
            raise


class ImageExtractor:
    """Extract and process images from PDFs for vision models"""
    
    def __init__(self):
        self.max_dimension = settings.IMAGE_MAX_DIMENSION
        self.quality = settings.IMAGE_COMPRESSION_QUALITY
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all images from PDF and generate descriptions
        """
        images = []
        
        try:
            # Convert PDF pages to images
            pdf_images = pdf2image.convert_from_path(
                pdf_path,
                first_page=1,
                last_page=min(settings.MAX_PAGES, 100)  # Limit to 100 pages for speed
            )
            
            for page_num, image in enumerate(pdf_images, 1):
                # Check if image has significant content (not just blank page)
                if self._is_content_rich(image):
                    processed = self._process_image(image)
                    images.append({
                        "page": page_num,
                        "image": processed["base64"],
                        "dimensions": processed["dimensions"],
                        "has_text": processed["has_text"],
                        "description": ""  # Will be filled by vision model
                    })
                    
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            
        return images
    
    def _process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Compress and prepare image for vision model
        """
        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
            
        # Resize if too large
        if max(image.size) > self.max_dimension:
            ratio = self.max_dimension / max(image.size)
            new_size = tuple(int(d * ratio) for d in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Try to detect if image has text
        has_text = self._detect_text(image)
        
        return {
            "base64": image_base64,
            "dimensions": image.size,
            "has_text": has_text
        }
    
    def _is_content_rich(self, image: Image.Image) -> bool:
        """Check if image has meaningful content (not blank)"""
        # Convert to grayscale and calculate variance
        gray = image.convert("L")
        array = np.array(gray)
        variance = np.var(array)
        
        # If variance is very low, it's likely a blank/white page
        return variance > 100
    
    def _detect_text(self, image: Image.Image) -> bool:
        """Detect if image contains readable text using OCR"""
        try:
            text = pytesseract.image_to_string(image)
            return len(text.strip()) > 10  # At least 10 chars of text
        except Exception:
            return False


class PDFParserFactory:
    """Factory for creating appropriate PDF parser"""
    
    _parsers = {
        "marker": MarkerPDFParser,
        "unstructured": UnstructuredPDFParser,
    }
    
    @classmethod
    def create_parser(cls, method: str = None) -> PDFParser:
        """Create parser based on configured method"""
        method = method or settings.PDF_EXTRACTION_METHOD
        
        if method not in cls._parsers:
            logger.warning(f"Parser {method} not found, using unstructured")
            method = "unstructured"
            
        return cls._parsers[method]()


def parse_pdf(pdf_path: str, extract_images: bool = True) -> Dict[str, Any]:
    """
    Main function to parse PDF with all features
    """
    parser = PDFParserFactory.create_parser()
    content = parser.parse(pdf_path)
    
    if extract_images and settings.EXTRACT_IMAGES:
        image_extractor = ImageExtractor()
        images = image_extractor.extract_images_from_pdf(pdf_path)
        content["images"] = images
    
    return content
