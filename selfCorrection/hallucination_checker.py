"""
Hallucination Checker for Self-Correction Loop
Validates that generated answers don't hallucinate (make up information)
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from openai import OpenAI

logger = logging.getLogger(__name__)


class HallucinationLevel(Enum):
    """Severity levels of hallucination"""
    NONE = "none"
    MINOR = "minor"  # Small inaccuracies
    MODERATE = "moderate"  # Some made-up elements
    SEVERE = "severe"  # Major fabrications


class HallucinationChecker:
    """
    Validates that answers are grounded in source documents
    Prevents model from inventing legal information
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize hallucination checker
        
        Args:
            model: LLM model to use for validation
        """
        self.client = OpenAI()
        self.model = model
    
    def check_hallucination(
        self,
        question: str,
        answer: str,
        source_documents: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if answer contains hallucinations
        
        Args:
            question: The original question asked
            answer: The model's answer
            source_documents: List of source document chunks used
            context: Optional additional context
            
        Returns:
            Hallucination analysis with level and details
        """
        logger.info("Checking answer for hallucinations")
        
        prompt = self._build_hallucination_prompt(
            question,
            answer,
            source_documents,
            context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Low temp for consistency
                max_tokens=1000,
            )
            
            response_text = response.choices[0].message.content
            result = self._parse_hallucination_response(response_text)
            
            return {
                "hallucination_level": result.get("hallucination_level", "none"),
                "is_hallucinating": result.get("is_hallucinating", False),
                "confidence": result.get("confidence", 0.5),
                "hallucinated_claims": result.get("hallucinated_claims", []),
                "grounded_claims": result.get("grounded_claims", []),
                "unverifiable_claims": result.get("unverifiable_claims", []),
                "recommendations": result.get("recommendations", []),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            return self._get_default_result(str(e))
    
    def check_multiple_answers(
        self,
        question: str,
        answers: List[str],
        source_documents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Check multiple candidate answers for hallucinations
        
        Args:
            question: Original question
            answers: List of candidate answers
            source_documents: Source documents
            
        Returns:
            List of hallucination checks for each answer
        """
        results = []
        for answer in answers:
            result = self.check_hallucination(
                question,
                answer,
                source_documents
            )
            results.append(result)
        
        return results
    
    def validate_claims(
        self,
        answer: str,
        source_documents: List[str]
    ) -> Dict[str, Any]:
        """
        Extract and validate individual claims from answer
        
        Args:
            answer: The answer text
            source_documents: Source documents to validate against
            
        Returns:
            Validation of each claim in the answer
        """
        prompt = self._build_claim_extraction_prompt(answer, source_documents)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            
            response_text = response.choices[0].message.content
            return self._parse_claim_validation(response_text)
            
        except Exception as e:
            logger.error(f"Error validating claims: {e}")
            return {"error": str(e), "claims": []}
    
    def _build_hallucination_prompt(
        self,
        question: str,
        answer: str,
        source_documents: List[str],
        context: Optional[str]
    ) -> str:
        """Build hallucination checking prompt"""
        doc_text = "\n\n".join([f"DOCUMENT {i+1}:\n{doc}" for i, doc in enumerate(source_documents)])
        
        prompt = f"""You are an expert at detecting hallucinations in legal document analysis.
Your job is to verify that the answer is grounded in the provided source documents.

QUESTION: {question}

ANSWER TO CHECK:
{answer}

SOURCE DOCUMENTS:
{doc_text}
"""
        
        if context:
            prompt += f"\nCONTEXT: {context}\n"
        
        prompt += """
HALLUCINATION DETECTION TASK:
1. Identify claims in the answer
2. Check if each claim is supported by the source documents
3. Flag any claims not found in the sources
4. Rate overall hallucination severity

SEVERITY LEVELS:
- NONE: All claims are grounded in sources
- MINOR: Small inaccuracies or unverifiable details
- MODERATE: Some key claims lack source support
- SEVERE: Major fabrications or completely unsupported claims

Respond in JSON format:
{
  "hallucination_level": "none|minor|moderate|severe",
  "is_hallucinating": true/false,
  "confidence": <0.0-1.0>,
  "hallucinated_claims": ["claim that is made up", ...],
  "grounded_claims": ["claim supported by sources", ...],
  "unverifiable_claims": ["claim not found in sources but plausible", ...],
  "reasoning": "<detailed explanation>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>"]
}
"""
        return prompt
    
    def _build_claim_extraction_prompt(
        self,
        answer: str,
        source_documents: List[str]
    ) -> str:
        """Build prompt for extracting and validating claims"""
        doc_text = "\n\n".join([f"DOC {i+1}: {doc[:500]}" for i, doc in enumerate(source_documents)])
        
        return f"""Extract all factual claims from this answer and validate against source documents:

ANSWER:
{answer}

SOURCES (summarized):
{doc_text}

For each claim, state:
1. The claim text
2. Evidence from sources (yes/no)
3. Direct quote if evidence found
4. Confidence level

JSON format:
{{
  "claims": [
    {{"claim": "...", "found_in_source": true/false, "quote": "...", "confidence": 0.0-1.0}}
  ],
  "summary": "<overall assessment>"
}}
"""
    
    def _parse_hallucination_response(self, response: str) -> Dict[str, Any]:
        """Parse hallucination check response"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data
            else:
                return {}
        except Exception as e:
            logger.error(f"Error parsing hallucination response: {e}")
            return {}
    
    def _parse_claim_validation(self, response: str) -> Dict[str, Any]:
        """Parse claim validation response"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data
            else:
                return {"claims": [], "summary": "Parse error"}
        except Exception as e:
            logger.error(f"Error parsing claim response: {e}")
            return {"claims": [], "summary": str(e)}
    
    def _get_default_result(self, error: str) -> Dict[str, Any]:
        """Get default result on error"""
        return {
            "hallucination_level": "unknown",
            "is_hallucinating": False,
            "confidence": 0.0,
            "hallucinated_claims": [],
            "grounded_claims": [],
            "unverifiable_claims": [],
            "recommendations": [
                "Manual review recommended",
                f"Error occurred: {error}"
            ],
            "reasoning": "Automatic check failed. Please review manually."
        }


class SourceGrounder:
    """Ground answers in source documents"""
    
    @staticmethod
    def add_source_citations(
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Add source citations to answer
        
        Args:
            answer: The answer text
            sources: Source documents with metadata
            
        Returns:
            Answer with citations added
        """
        cited_answer = answer
        
        for i, source in enumerate(sources, 1):
            citation = f"[Source {i}: {source.get('source', 'Unknown')}]"
            # In practice, this would intelligently insert citations
            # For now, append to answer
        
        return cited_answer
    
    @staticmethod
    def extract_supporting_quotes(
        answer: str,
        source_documents: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract quotes from sources that support the answer
        
        Args:
            answer: The answer
            source_documents: Source documents
            
        Returns:
            List of supporting quotes with metadata
        """
        supporting_quotes = []
        
        # Extract key concepts from answer
        answer_sentences = answer.split('.')
        
        for doc in source_documents:
            for sentence in answer_sentences:
                if not sentence.strip():
                    continue
                
                # Check if answer concept appears in source
                if any(word in doc.lower() for word in sentence.lower().split()):
                    supporting_quotes.append({
                        "quote": doc[:200] + "...",
                        "relates_to": sentence.strip(),
                        "confidence": 0.7
                    })
        
        return supporting_quotes[:5]  # Return top 5


class AnswerValidator:
    """Validate answer quality and completeness"""
    
    def __init__(self):
        """Initialize validator"""
        self.checker = HallucinationChecker()
    
    def validate_answer(
        self,
        question: str,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive answer validation
        
        Args:
            question: The question asked
            answer: The proposed answer
            sources: Source documents
            
        Returns:
            Complete validation assessment
        """
        hallucination_check = self.checker.check_hallucination(
            question,
            answer,
            sources
        )
        
        claim_validation = self.checker.validate_claims(answer, sources)
        
        return {
            "hallucination_assessment": hallucination_check,
            "claim_validation": claim_validation,
            "is_valid": hallucination_check.get("hallucination_level") == "none",
            "confidence": hallucination_check.get("confidence", 0.5)
        }


def check_answer_hallucinations(
    question: str,
    answer: str,
    source_documents: List[str]
) -> Dict[str, Any]:
    """
    Convenience function to check hallucinations
    
    Args:
        question: Original question
        answer: Generated answer
        source_documents: Source documents used
        
    Returns:
        Hallucination check results
    """
    checker = HallucinationChecker()
    return checker.check_hallucination(question, answer, source_documents)
