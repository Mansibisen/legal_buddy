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
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
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
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF using Marker with markdown preservation
        """
        try:
            logger.info(f"Parsing PDF with Marker: {pdf_path}")
            
            # Use unstructured library (which can use marker backend)
            elements = partition_pdf(
                pdf_path,
                languages=["en"],
                strategy="hi_res",  # High resolution parsing
                extract_image_block_types=["Image", "Figure"],
                infer_table_structure=True,
            )
            
            return self._process_elements(elements, pdf_path)
            
        except Exception as e:
            logger.error(f"Error parsing PDF with Marker: {e}")
            raise

    def _process_elements(self, elements: List, pdf_path: str) -> Dict[str, Any]:
        """Process unstructured elements into organized content"""
        content = {
            "text_content": "",
            "sections": [],
            "images": [],
            "tables": [],
            "metadata": {
                "source": pdf_path,
                "parser": self.name,
            }
        }
        
        current_section = None
        
        for element in elements:
            element_type = type(element).__name__
            
            if element_type in ["Title", "Heading"]:
                if current_section:
                    content["sections"].append(current_section)
                current_section = {
                    "title": str(element),
                    "content": "",
                    "subsections": []
                }
                
            elif element_type == "NarrativeText":
                if current_section is not None:
                    current_section["content"] += str(element) + "\n"
                else:
                    content["text_content"] += str(element) + "\n"
                    
            elif element_type == "Table":
                table_data = {
                    "html": element.metadata.text_as_html if hasattr(element.metadata, "text_as_html") else str(element),
                    "text": str(element)
                }
                content["tables"].append(table_data)
                
            elif element_type == "Image":
                image_data = {
                    "source": str(element),
                    "description": ""
                }
                content["images"].append(image_data)
                
        if current_section:
            content["sections"].append(current_section)
            
        return content


class UnstructuredPDFParser(PDFParser):
    """
    Uses Unstructured.io library for PDF parsing
    Good for general document processing
    """
    
    def __init__(self):
        self.name = "unstructured"
        
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using Unstructured library"""
        try:
            logger.info(f"Parsing PDF with Unstructured: {pdf_path}")
            
            elements = partition(
                pdf_path,
                include_page_breaks=True,
                strategy="hi_res",
            )
            
            content = {
                "text_content": "\n".join([str(e) for e in elements]),
                "elements": [str(e) for e in elements],
                "metadata": {"source": pdf_path, "parser": self.name}
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Error parsing PDF with Unstructured: {e}")
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
