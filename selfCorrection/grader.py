"""
Document Grader for Legal Retrieval
Validates if retrieved documents actually answer the question
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from openai import OpenAI

logger = logging.getLogger(__name__)


class GradeScore(Enum):
    """Grade scores for document relevance"""
    RELEVANT = "relevant"          # Document answers the question
    PARTIALLY_RELEVANT = "partial" # Document partially answers
    IRRELEVANT = "irrelevant"      # Document doesn't answer


class DocumentGrader:
    """
    Grades whether retrieved documents answer the user's question
    Uses Claude/GPT-4 to perform intelligent relevance assessment
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize document grader
        
        Args:
            model: LLM model to use for grading
        """
        self.client = OpenAI()
        self.model = model
        
    def grade_document(
        self,
        question: str,
        document: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Grade a single document for relevance to question
        
        Args:
            question: User question
            document: Retrieved document text
            context: Optional context about the document
            
        Returns:
            Grade assessment with score and reasoning
        """
        logger.info(f"Grading document relevance for: {question[:50]}...")
        
        prompt = self._build_grading_prompt(question, document, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=500,
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            result = self._parse_grade_response(response_text, question, document)
            
            return result
            
        except Exception as e:
            logger.error(f"Error grading document: {e}")
            return {
                "grade": GradeScore.IRRELEVANT.value,
                "score": 0.0,
                "reasoning": f"Grading error: {str(e)}",
                "question": question,
                "document_preview": document[:200]
            }
    
    def grade_documents(
        self,
        question: str,
        documents: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Grade multiple documents
        
        Args:
            question: User question
            documents: List of documents to grade
            context: Optional context
            
        Returns:
            List of grade assessments
        """
        logger.info(f"Grading {len(documents)} documents")
        
        results = []
        for doc in documents:
            result = self.grade_document(question, doc, context)
            results.append(result)
        
        return results
    
    def filter_relevant_documents(
        self,
        question: str,
        documents: List[str],
        threshold: str = "relevant"
    ) -> List[str]:
        """
        Filter documents by relevance threshold
        
        Args:
            question: User question
            documents: List of documents
            threshold: Minimum grade ("relevant", "partial", "irrelevant")
            
        Returns:
            Filtered list of relevant documents
        """
        grades = self.grade_documents(question, documents)
        
        # Grade ordering
        grade_order = {
            GradeScore.RELEVANT.value: 3,
            GradeScore.PARTIALLY_RELEVANT.value: 2,
            GradeScore.IRRELEVANT.value: 1
        }
        
        threshold_value = grade_order.get(threshold, 2)
        
        relevant = [
            doc for doc, grade in zip(documents, grades)
            if grade_order.get(grade["grade"], 1) >= threshold_value
        ]
        
        logger.info(f"Filtered to {len(relevant)}/{len(documents)} relevant documents")
        return relevant
    
    def _build_grading_prompt(
        self,
        question: str,
        document: str,
        context: Optional[str] = None
    ) -> str:
        """Build grading prompt"""
        prompt = f"""You are an expert legal document grader. Your task is to assess whether 
a retrieved document answers a user's question about a legal contract or agreement.

QUESTION: {question}

DOCUMENT: {document}
"""
        
        if context:
            prompt += f"\nCONTEXT: {context}\n"
        
        prompt += """
Please grade this document on the following scale:
- RELEVANT: The document clearly answers the question or provides the information needed
- PARTIAL: The document partially addresses the question but lacks some details
- IRRELEVANT: The document does not answer the question

Respond in JSON format:
{
  "grade": "RELEVANT|PARTIAL|IRRELEVANT",
  "score": <0.0-1.0>,
  "reasoning": "<brief explanation>",
  "key_phrases": ["<phrases from document that answer the question>"],
  "missing_info": "<any missing information if partial>"
}
"""
        return prompt
    
    def _parse_grade_response(
        self,
        response: str,
        question: str,
        document: str
    ) -> Dict[str, Any]:
        """Parse grader response"""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                grade_data = json.loads(json_match.group())
            else:
                grade_data = {}
            
            # Normalize grade
            grade_raw = grade_data.get("grade", "IRRELEVANT").upper()
            if "RELEVANT" in grade_raw and "PARTIAL" not in grade_raw:
                grade = GradeScore.RELEVANT.value
            elif "PARTIAL" in grade_raw:
                grade = GradeScore.PARTIALLY_RELEVANT.value
            else:
                grade = GradeScore.IRRELEVANT.value
            
            return {
                "grade": grade,
                "score": float(grade_data.get("score", 0.0)),
                "reasoning": grade_data.get("reasoning", ""),
                "key_phrases": grade_data.get("key_phrases", []),
                "missing_info": grade_data.get("missing_info", ""),
                "question": question,
                "document_preview": document[:300]
            }
            
        except Exception as e:
            logger.error(f"Error parsing grade response: {e}")
            return {
                "grade": GradeScore.IRRELEVANT.value,
                "score": 0.0,
                "reasoning": f"Parse error: {str(e)}",
                "question": question,
                "document_preview": document[:300]
            }


class DocumentRelevanceValidator:
    """Additional validation for document relevance"""
    
    @staticmethod
    def check_keyword_coverage(
        question: str,
        document: str,
        required_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check if document covers key terms from question
        
        Args:
            question: User question
            document: Document text
            required_keywords: Optional list of keywords that must appear
            
        Returns:
            Coverage analysis
        """
        doc_lower = document.lower()
        question_lower = question.lower()
        
        # Extract key terms from question
        import re
        key_terms = re.findall(r'\b\w{4,}\b', question_lower)
        key_terms = [t for t in key_terms if len(t) > 3]
        
        # Use provided keywords if available
        if required_keywords:
            key_terms = required_keywords
        
        # Check coverage
        covered = [term for term in key_terms if term in doc_lower]
        coverage_ratio = len(covered) / len(key_terms) if key_terms else 0
        
        return {
            "total_terms": len(key_terms),
            "covered_terms": len(covered),
            "coverage_ratio": coverage_ratio,
            "covered": covered,
            "missing": [t for t in key_terms if t not in covered]
        }
    
    @staticmethod
    def check_clause_presence(
        document: str,
        clause_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check if specific legal clauses are present
        
        Args:
            document: Document text
            clause_patterns: Patterns to search for
            
        Returns:
            Clause presence analysis
        """
        if not clause_patterns:
            clause_patterns = [
                "payment",
                "liability",
                "indemnif",
                "confidential",
                "termination",
                "warranty",
                "force majeure"
            ]
        
        doc_lower = document.lower()
        found_clauses = [p for p in clause_patterns if p in doc_lower]
        
        return {
            "found_clauses": found_clauses,
            "total_clauses": len(clause_patterns),
            "clause_ratio": len(found_clauses) / len(clause_patterns) if clause_patterns else 0
        }
    
    @staticmethod
    def assess_document_quality(
        document: str,
        min_length: int = 100,
        min_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Assess overall document quality
        
        Args:
            document: Document text
            min_length: Minimum character length
            min_sentences: Minimum sentence count
            
        Returns:
            Quality assessment
        """
        import re
        
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        quality_score = 1.0
        issues = []
        
        if len(document) < min_length:
            quality_score *= 0.7
            issues.append(f"Document too short ({len(document)} chars)")
        
        if len(sentences) < min_sentences:
            quality_score *= 0.8
            issues.append(f"Too few sentences ({len(sentences)})")
        
        # Check for specific legal content indicators
        if any(word in document.lower() for word in ["section", "article", "clause", "paragraph"]):
            quality_score *= 1.1  # Boost for structured legal content
        
        quality_score = min(1.0, quality_score)
        
        return {
            "quality_score": quality_score,
            "length": len(document),
            "sentence_count": len(sentences),
            "issues": issues,
            "is_quality": quality_score >= 0.7
        }


def grade_retrieval_results(
    question: str,
    documents: List[str],
    min_relevant: int = 1
) -> Dict[str, Any]:
    """
    Convenience function to grade retrieval results
    
    Args:
        question: User question
        documents: Retrieved documents
        min_relevant: Minimum number of relevant documents needed
        
    Returns:
        Grading results and assessment
    """
    grader = DocumentGrader()
    
    grades = grader.grade_documents(question, documents)
    
    relevant_count = sum(1 for g in grades if g["grade"] == GradeScore.RELEVANT.value)
    partial_count = sum(1 for g in grades if g["grade"] == GradeScore.PARTIALLY_RELEVANT.value)
    
    return {
        "total_documents": len(documents),
        "relevant_count": relevant_count,
        "partial_count": partial_count,
        "irrelevant_count": len(documents) - relevant_count - partial_count,
        "grades": grades,
        "has_sufficient_evidence": relevant_count >= min_relevant,
        "relevance_ratio": relevant_count / len(documents) if documents else 0
    }
