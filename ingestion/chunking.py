"""
Section-Aware Chunking for Legal Documents
Intelligently splits documents by legal clauses and sections
instead of simple character-based splitting
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    document_id: Optional[str] = None
    chunk_type: str = "text"  # text, table, image_description
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens (1 token ≈ 4 chars)"""
        return len(self.content) // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "content": self.content,
            "section_title": self.section_title,
            "section_number": self.section_number,
            "metadata": {
                **self.metadata,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "chunk_type": self.chunk_type,
                "document_id": self.document_id,
            }
        }


class Chunker(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks"""
        pass


class CharacterBasedChunker(Chunker):
    """
    Simple character-based chunking with overlap
    Fallback for unstructured text
    """
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text by character count"""
        chunks = []
        
        if not text.strip():
            return chunks
        
        lines = text.split("\n")
        current_chunk = ""
        start_line = 0
        
        for line_num, line in enumerate(lines):
            # Add line to current chunk
            potential_chunk = current_chunk + "\n" + line if current_chunk else line
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current line would exceed limit
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        start_line=start_line,
                        end_line=line_num - 1,
                        metadata=metadata or {}
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + "\n" + line
                start_line = max(0, line_num - len(overlap_lines.split("\n")) + 1)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                start_line=start_line,
                end_line=len(lines) - 1,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_lines(self, text: str) -> str:
        """Get last N lines for overlap"""
        lines = text.split("\n")
        overlap_count = max(1, len(lines) // 3)  # ~33% overlap
        return "\n".join(lines[-overlap_count:])


class SectionAwareChunker(Chunker):
    """
    Legal document-aware chunking
    Splits by sections, subsections, and legal structures
    """
    
    # Patterns for legal section headers
    SECTION_PATTERNS = [
        (r"^(?:Article|Section|Sec\.?)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)$", "section"),
        (r"^(Chapter|Part|Clause)\s+(\d+)[:\s]+(.*?)$", "chapter"),
        (r"^(\d+\.?\d*)\s+([A-Z][^:]*)[:\s]", "numbered"),
        (r"^(#{1,6})\s+(.*?)$", "markdown"),
    ]
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), type_name)
            for pattern, type_name in self.SECTION_PATTERNS
        ]
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text by legal sections
        """
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Parse document structure
        sections = self._parse_sections(text)
        
        # Create chunks for each section
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse document into sections"""
        sections = []
        lines = text.split("\n")
        
        current_section = {
            "title": "Preamble",
            "number": "",
            "content": [],
            "start_line": 0,
            "end_line": 0,
        }
        
        for line_num, line in enumerate(lines):
            # Check if line is a section header
            section_match = self._match_section_header(line)
            
            if section_match:
                # Save previous section
                if current_section["content"]:
                    current_section["end_line"] = line_num - 1
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": section_match["title"],
                    "number": section_match["number"],
                    "content": [line],
                    "start_line": line_num,
                    "end_line": line_num,
                }
            else:
                current_section["content"].append(line)
                current_section["end_line"] = line_num
        
        # Add final section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _match_section_header(self, line: str) -> Optional[Dict[str, str]]:
        """Check if line is a section header"""
        for pattern, pattern_type in self.compiled_patterns:
            match = pattern.match(line.strip())
            if match:
                groups = match.groups()
                
                if pattern_type == "section":
                    return {
                        "number": groups[0],
                        "title": groups[1],
                        "type": "section"
                    }
                elif pattern_type == "chapter":
                    return {
                        "number": groups[1],
                        "title": f"{groups[0]} {groups[1]}: {groups[2]}",
                        "type": "chapter"
                    }
                elif pattern_type == "numbered":
                    return {
                        "number": groups[0],
                        "title": groups[1],
                        "type": "numbered"
                    }
                elif pattern_type == "markdown":
                    return {
                        "number": str(len(groups[0])),
                        "title": groups[1],
                        "type": "markdown"
                    }
        
        return None
    
    def _chunk_section(
        self,
        section: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Break section into smaller chunks if needed"""
        section_text = "\n".join(section["content"])
        chunks = []
        
        if len(section_text) <= self.chunk_size:
            # Section fits in one chunk
            chunk = Chunk(
                content=section_text,
                section_title=section["title"],
                section_number=section["number"],
                start_line=section["start_line"],
                end_line=section["end_line"],
                metadata=metadata or {}
            )
            chunks.append(chunk)
        else:
            # Break section into multiple chunks
            sub_chunks = self._split_long_section(
                section_text,
                section["title"],
                section["number"],
                section["start_line"]
            )
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_long_section(
        self,
        text: str,
        section_title: str,
        section_number: str,
        start_line: int
    ) -> List[Chunk]:
        """Split a long section using character-based chunking"""
        chunks = []
        lines = text.split("\n")
        current_chunk = ""
        chunk_start_line = start_line
        
        for line_offset, line in enumerate(lines):
            potential_chunk = current_chunk + "\n" + line if current_chunk else line
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunk = Chunk(
                        content=current_chunk.strip(),
                        section_title=section_title,
                        section_number=section_number,
                        start_line=chunk_start_line,
                        end_line=chunk_start_line + len(current_chunk.split("\n")) - 1,
                        metadata={"is_subsection": True}
                    )
                    chunks.append(chunk)
                
                # Prepare overlap for next chunk
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + "\n" + line
                chunk_start_line = start_line + line_offset - len(overlap_text.split("\n")) + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                section_title=section_title,
                section_number=section_number,
                start_line=chunk_start_line,
                end_line=start_line + len(lines) - 1,
                metadata={"is_subsection": True}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get text for overlap between chunks"""
        lines = text.split("\n")
        overlap_lines = max(2, len(lines) // 5)  # ~20% overlap
        return "\n".join(lines[-overlap_lines:])


class HybridChunker(Chunker):
    """
    Combines section-aware and character-based chunking
    Uses section-aware by default, falls back to character-based for unstructured sections
    """
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.section_aware = SectionAwareChunker(chunk_size, overlap)
        self.character_based = CharacterBasedChunker(chunk_size, overlap)
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Intelligently chunk text
        """
        # Try section-aware first
        try:
            chunks = self.section_aware.chunk(text, metadata)
            if chunks:
                logger.info(f"Created {len(chunks)} section-aware chunks")
                return chunks
        except Exception as e:
            logger.warning(f"Section-aware chunking failed: {e}, falling back to character-based")
        
        # Fall back to character-based
        chunks = self.character_based.chunk(text, metadata)
        logger.info(f"Created {len(chunks)} character-based chunks")
        return chunks


def chunk_document(
    document: Dict[str, Any],
    chunker: Optional[Chunker] = None,
    include_images: bool = True
) -> List[Chunk]:
    """
    Main function to chunk entire document
    
    Args:
        document: Parsed document with content
        chunker: Chunker to use (default: HybridChunker)
        include_images: Include image descriptions as chunks
        
    Returns:
        List of chunks
    """
    if chunker is None:
        chunker = HybridChunker()
    
    chunks = []
    
    # Chunk main text content
    if "text_content" in document and document["text_content"].strip():
        text_chunks = chunker.chunk(
            document["text_content"],
            metadata={"source": document.get("metadata", {}).get("source", "")}
        )
        chunks.extend(text_chunks)
    
    # Chunk sections
    if "sections" in document:
        for section in document["sections"]:
            section_text = f"{section['title']}\n\n{section['content']}"
            section_chunks = chunker.chunk(
                section_text,
                metadata={
                    "section": section["title"],
                    "source": document.get("metadata", {}).get("source", "")
                }
            )
            for chunk in section_chunks:
                chunk.section_title = section["title"]
            chunks.extend(section_chunks)
    
    # Include image descriptions as chunks
    if include_images and "images" in document:
        for image in document["images"]:
            if image.get("description"):
                img_chunk = Chunk(
                    content=f"[Image from page {image['page']}]\n{image['description']}",
                    chunk_type="image_description",
                    metadata={
                        "page": image["page"],
                        "source": document.get("metadata", {}).get("source", "")
                    }
                )
                chunks.append(img_chunk)
    
    logger.info(f"Document chunked into {len(chunks)} chunks")
    return chunks
