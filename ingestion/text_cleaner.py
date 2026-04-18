"""
Text Cleaning and Preprocessing for Legal Documents
Handles boilerplate removal, normalization, and legal-specific cleaning
"""
import logging
import re
from typing import List, Dict, Any, Optional, Pattern
from abc import ABC, abstractmethod

import unicodedata

from config import settings

logger = logging.getLogger(__name__)


class TextCleaner(ABC):
    """Abstract base class for text cleaners"""
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text and return result"""
        pass


class BoilerplateCleaner(TextCleaner):
    """Remove common legal boilerplate text"""
    
    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        # Footer/header patterns
        r"Page \d+.*?(?=\n|$)",
        r"^\s*-+\s*$",
        
        # Common legal footers
        r"Confidential - .*?(?=\n|$)",
        r"Privileged and Confidential.*?(?=\n|$)",
        r"Copyright \d{4}.*?All Rights Reserved.*?(?=\n|$)",
        
        # Electronic signature disclaimers
        r"This document was digitally signed.*?(?=\n{2}|$)",
        r"Electronic signature.*?(?=\n{2}|$)",
        
        # Common "TO" address blocks
        r"^TO:.*?$",
        r"^FROM:.*?$",
        
        # Page number indicators
        r"Page [0-9ivxIVX]+ (of|/) [0-9ivxIVX]+",
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.compiled_patterns: List[Pattern] = [
            re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            for pattern in self.BOILERPLATE_PATTERNS
        ]
    
    def clean(self, text: str) -> str:
        """Remove boilerplate patterns"""
        if not self.enabled:
            return text
        
        cleaned = text
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub("", cleaned)
        
        return cleaned


class PageNumberCleaner(TextCleaner):
    """Remove page numbers and page breaks"""
    
    PATTERNS = [
        r"\n\s*\d+\s*\n",  # Standalone page numbers
        r"^Page \d+$",
        r"^- \d+ -$",
        r"\f",  # Form feed character
        r"\x0c",  # Page break
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.compiled_patterns: List[Pattern] = [
            re.compile(pattern, re.MULTILINE)
            for pattern in self.PATTERNS
        ]
    
    def clean(self, text: str) -> str:
        """Remove page numbers and breaks"""
        if not self.enabled:
            return text
        
        cleaned = text
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub("\n", cleaned)
        
        return cleaned


class WhitespaceNormalizer(TextCleaner):
    """Normalize whitespace while preserving structure"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def clean(self, text: str) -> str:
        """Normalize whitespace"""
        if not self.enabled:
            return text
        
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        
        # Replace multiple newlines with double newline (preserve paragraph structure)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        
        # Remove trailing whitespace from lines
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class SpecialCharacterCleaner(TextCleaner):
    """Handle special characters and encoding issues"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def clean(self, text: str) -> str:
        """Clean special characters"""
        if not self.enabled:
            return text
        
        # Handle common smart quotes
        replacements = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "–": "-",
            "—": "-",
            "…": "...",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove control characters except newline and tab
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
        
        return text


class LegalSpecificCleaner(TextCleaner):
    """Legal document-specific cleaning"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def clean(self, text: str) -> str:
        """Apply legal-specific cleaning rules"""
        if not self.enabled:
            return text
        
        # Normalize section numbering
        # "§ 1.1" or "Sec. 1.1" -> "Section 1.1"
        text = re.sub(r"§\s*(\d+\.?\d*)", r"Section \1", text)
        text = re.sub(r"\bSec\.\s*(\d+\.?\d*)", r"Section \1", text)
        text = re.sub(r"\bSubsec\.\s*(\d+\.?\d*)", r"Subsection \1", text)
        
        # Normalize case names: "Smith v. Jones" format
        text = re.sub(r"\bv\.\s+", r"v. ", text)
        
        # Normalize citation format (basic)
        # "[1]" -> "[1]" (keep but clean)
        text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
        
        # Remove excessive highlighting/redaction markers
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"_{3,}", "", text)
        
        return text


class ComprehensiveTextCleaner:
    """
    Comprehensive text cleaner combining multiple strategies
    """
    
    def __init__(self, config: Optional[Dict[str, bool]] = None):
        """
        Initialize cleaner with optional config
        
        Args:
            config: Dict with keys like "boilerplate", "page_numbers", etc.
        """
        # Use settings or provided config
        config = config or {
            "boilerplate": settings.REMOVE_BOILERPLATE,
            "page_numbers": settings.REMOVE_PAGE_NUMBERS,
            "whitespace": settings.NORMALIZE_WHITESPACE,
            "special_chars": True,
            "legal_specific": True,
        }
        
        self.cleaners: List[TextCleaner] = [
            SpecialCharacterCleaner(enabled=config.get("special_chars", True)),
            BoilerplateCleaner(enabled=config.get("boilerplate", True)),
            PageNumberCleaner(enabled=config.get("page_numbers", True)),
            LegalSpecificCleaner(enabled=config.get("legal_specific", True)),
            WhitespaceNormalizer(enabled=config.get("whitespace", True)),
        ]
    
    def clean(self, text: str) -> str:
        """Apply all cleaning steps"""
        cleaned = text
        
        for cleaner in self.cleaners:
            cleaned = cleaner.clean(cleaned)
        
        return cleaned
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts"""
        return [self.clean(text) for text in texts]


class DocumentCleaner:
    """Clean entire parsed document structure"""
    
    def __init__(self):
        self.text_cleaner = ComprehensiveTextCleaner()
    
    def clean_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all text content in document
        """
        logger.info("Cleaning document")
        
        cleaned_doc = doc.copy()
        
        # Clean main text content
        if "text_content" in cleaned_doc:
            cleaned_doc["text_content"] = self.text_cleaner.clean(cleaned_doc["text_content"])
        
        # Clean sections
        if "sections" in cleaned_doc:
            for section in cleaned_doc["sections"]:
                section["title"] = self.text_cleaner.clean(section["title"])
                section["content"] = self.text_cleaner.clean(section["content"])
        
        # Clean table text
        if "tables" in cleaned_doc:
            for table in cleaned_doc["tables"]:
                if "text" in table:
                    table["text"] = self.text_cleaner.clean(table["text"])
        
        return cleaned_doc


def clean_text(text: str, config: Optional[Dict[str, bool]] = None) -> str:
    """
    Convenience function to clean text
    """
    cleaner = ComprehensiveTextCleaner(config)
    return cleaner.clean(text)


def clean_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to clean entire document
    """
    cleaner = DocumentCleaner()
    return cleaner.clean_document(doc)
