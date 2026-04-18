"""
Multi-Modal Answer Generator with Citations
Generates comprehensive answers with image context and citations
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation information for answer"""
    source_id: str
    chunk_id: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    quote: Optional[str] = None
    confidence: float = 0.9


@dataclass
class MultiModalAnswer:
    """Complete answer with multiple modalities"""
    answer_text: str
    citations: List[Citation]
    source_documents: List[str]
    supporting_images: List[Dict[str, Any]] = None
    confidence: float = 0.8
    generated_at: str = None
    model_used: str = ""
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now().isoformat()
        if self.supporting_images is None:
            self.supporting_images = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer_text,
            "citations": [
                {
                    "source_id": c.source_id,
                    "chunk_id": c.chunk_id,
                    "page": c.page_number,
                    "section": c.section,
                    "clause": c.clause,
                    "quote": c.quote,
                    "confidence": c.confidence
                }
                for c in self.citations
            ],
            "sources": self.source_documents,
            "supporting_images": self.supporting_images,
            "confidence": self.confidence,
            "generated_at": self.generated_at,
            "model": self.model_used
        }


class MultiModalAnswerGenerator:
    """
    Generates comprehensive answers with text, images, and citations
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize answer generator
        
        Args:
            model: LLM model to use
        """
        self.client = OpenAI()
        self.model = model
    
    def generate_answer(
        self,
        question: str,
        documents: List[str],
        document_metadata: List[Dict[str, Any]] = None,
        include_images: bool = True
    ) -> MultiModalAnswer:
        """
        Generate comprehensive answer with citations
        
        Args:
            question: Question to answer
            documents: Source document chunks
            document_metadata: Metadata for documents
            include_images: Whether to include supporting images
            
        Returns:
            MultiModalAnswer with citations and supporting content
        """
        logger.info(f"Generating answer for question: {question}")
        
        # Prepare context with images if available
        context_with_images = self._prepare_context(
            documents,
            document_metadata,
            include_images
        )
        
        # Generate answer
        prompt = self._build_generation_prompt(
            question,
            context_with_images["text_content"],
            context_with_images["has_images"]
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for factuality
                max_tokens=1500,
            )
            
            answer_text = response.choices[0].message.content
            
            # Extract citations
            citations = self._extract_citations(
                answer_text,
                documents,
                document_metadata
            )
            
            # Build multi-modal answer
            multi_modal_answer = MultiModalAnswer(
                answer_text=answer_text,
                citations=citations,
                source_documents=documents,
                supporting_images=context_with_images.get("images", []),
                model_used=self.model
            )
            
            return multi_modal_answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return MultiModalAnswer(
                answer_text=f"Error generating answer: {str(e)}",
                citations=[],
                source_documents=documents,
                model_used=self.model
            )
    
    def generate_with_context_injection(
        self,
        question: str,
        documents: List[str],
        document_metadata: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with explicit context injection
        
        Args:
            question: Question to answer
            documents: Source chunks
            document_metadata: Document metadata
            
        Returns:
            Answer with detailed context
        """
        # Prepare context
        context_text = "\n\n".join([
            f"[DOC {i}]\n{doc}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Answer this question based ONLY on the provided context.
Be concise but comprehensive. Include relevant details and numbers.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "context_documents": documents,
                "context_length": len(context_text),
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error in context injection: {e}")
            return {"error": str(e), "answer": ""}
    
    def generate_answer_with_images(
        self,
        question: str,
        text_documents: List[str],
        image_descriptions: List[str] = None,
        document_metadata: List[Dict[str, Any]] = None
    ) -> MultiModalAnswer:
        """
        Generate answer with explicit image integration
        
        Args:
            question: Question to answer
            text_documents: Text document chunks
            image_descriptions: Descriptions of relevant images
            document_metadata: Document metadata
            
        Returns:
            Answer with image context integrated
        """
        # Build prompt with both text and image context
        context = "TEXT CONTENT:\n" + "\n\n".join(text_documents)
        
        if image_descriptions:
            context += "\n\nIMAGE DESCRIPTIONS:\n" + "\n".join([
                f"- {desc}"
                for desc in image_descriptions
            ])
        
        prompt = f"""Answer this question using both text and image context:

{context}

QUESTION: {question}

Provide a comprehensive answer that integrates information from both text and images when relevant.
ANSWER:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            
            answer_text = response.choices[0].message.content
            
            # Create multi-modal answer
            return MultiModalAnswer(
                answer_text=answer_text,
                citations=[],
                source_documents=text_documents,
                supporting_images=[
                    {"description": desc}
                    for desc in (image_descriptions or [])
                ],
                model_used=self.model
            )
            
        except Exception as e:
            logger.error(f"Error generating image-integrated answer: {e}")
            return MultiModalAnswer(
                answer_text=f"Error: {str(e)}",
                citations=[],
                source_documents=text_documents,
                model_used=self.model
            )
    
    def _prepare_context(
        self,
        documents: List[str],
        metadata: List[Dict[str, Any]] = None,
        include_images: bool = True
    ) -> Dict[str, Any]:
        """Prepare context with potential images"""
        text_content = "\n\n".join([
            f"[DOCUMENT {i}]\n{doc}"
            for i, doc in enumerate(documents)
        ])
        
        images = []
        if include_images and metadata:
            for meta in metadata:
                if meta.get("has_images"):
                    images.append({
                        "source": meta.get("source"),
                        "page": meta.get("page_number"),
                        "description": meta.get("image_description", "Image")
                    })
        
        return {
            "text_content": text_content,
            "images": images,
            "has_images": len(images) > 0
        }
    
    def _build_generation_prompt(
        self,
        question: str,
        context: str,
        has_images: bool
    ) -> str:
        """Build answer generation prompt"""
        prompt = f"""You are an expert legal document analyst.
Answer the following question based ONLY on the provided context.
Be thorough, accurate, and cite specific sections when applicable.
"""
        
        if has_images:
            prompt += "\nNote: Some documents may have images with important information. "
            prompt += "Consider visual context when answering.\n"
        
        prompt += f"""
CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
        return prompt
    
    def _extract_citations(
        self,
        answer: str,
        documents: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> List[Citation]:
        """Extract citations from answer"""
        citations = []
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Simple citation extraction - finds matches between answer and documents
        for i, doc in enumerate(documents):
            # Check if content from this document appears in answer
            doc_sentences = doc.split('.')
            
            for sentence in doc_sentences:
                if len(sentence.strip()) > 20:  # Skip short fragments
                    # Check if substantial portion appears in answer
                    if sentence.strip()[:50] in answer:
                        citation = Citation(
                            source_id=metadata[i].get("source", f"Document {i+1}"),
                            chunk_id=f"chunk_{i}",
                            page_number=metadata[i].get("page_number"),
                            section=metadata[i].get("section"),
                            clause=metadata[i].get("clause"),
                            quote=sentence.strip()[:100],
                            confidence=0.85
                        )
                        citations.append(citation)
                        break
        
        return citations[:5]  # Return top 5 citations


class AnswerFormatter:
    """Format answers for different output formats"""
    
    @staticmethod
    def to_markdown(answer: MultiModalAnswer) -> str:
        """Format answer as Markdown"""
        output = f"# Answer\n\n{answer.answer_text}\n\n"
        
        if answer.citations:
            output += "## Sources\n\n"
            for i, citation in enumerate(answer.citations, 1):
                output += f"{i}. **{citation.source_id}**"
                if citation.section:
                    output += f" - {citation.section}"
                if citation.page_number:
                    output += f" (Page {citation.page_number})"
                output += "\n"
        
        if answer.supporting_images:
            output += "\n## Supporting Visuals\n\n"
            for img in answer.supporting_images:
                output += f"- {img.get('description', 'Image')}\n"
        
        return output
    
    @staticmethod
    def to_json(answer: MultiModalAnswer) -> Dict[str, Any]:
        """Format answer as JSON"""
        return answer.to_dict()
    
    @staticmethod
    def to_html(answer: MultiModalAnswer) -> str:
        """Format answer as HTML"""
        html = f"""
<div class="legal-answer">
    <h2>Answer</h2>
    <p>{answer.answer_text}</p>
    
    <h3>Sources</h3>
    <ul>
"""
        
        for citation in answer.citations:
            html += f"<li><strong>{citation.source_id}</strong>"
            if citation.section:
                html += f" - {citation.section}"
            if citation.page_number:
                html += f" (Page {citation.page_number})"
            html += "</li>\n"
        
        html += """
    </ul>
</div>
"""
        return html


def generate_legal_answer(
    question: str,
    documents: List[str],
    document_metadata: List[Dict[str, Any]] = None
) -> MultiModalAnswer:
    """
    Convenience function to generate answer
    
    Args:
        question: Question to answer
        documents: Source documents
        document_metadata: Document metadata
        
    Returns:
        MultiModalAnswer with citations
    """
    generator = MultiModalAnswerGenerator()
    return generator.generate_answer(
        question,
        documents,
        document_metadata
    )
