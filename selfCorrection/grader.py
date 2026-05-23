"""
Document Grader for Legal Retrieval
Validates if retrieved documents actually answer the question
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from langchain_ollama import OllamaLLM
from config import settings

logger = logging.getLogger(__name__)


class GradeScore(Enum):
    """Grade scores for document relevance"""
    RELEVANT = "relevant"          # Document answers the question
    PARTIALLY_RELEVANT = "partial" # Document partially answers
    IRRELEVANT = "irrelevant"      # Document doesn't answer


class DocumentGrader:
    """
    Grades whether retrieved documents answer the user's question
    Uses Llama to perform intelligent relevance assessment
    """
    
    def __init__(self, model: str = None):
        """
        Initialize document grader
        
        Args:
            model: LLM model to use for grading
        """
        self.model = model or settings.OLLAMA_MODEL
        self.client = OllamaLLM(
            model=self.model,
            base_url=settings.OLLAMA_BASE_URL
        )
        
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
            response_text = self.client.invoke(prompt)
            
            # Parse response
            result = self._parse_grade_response(response_text, question, document)
            
            return result
            
        except Exception as e:
            logger.error(f"Error grading document with LLM: {e}")
            # Fallback to keyword matching when LLM is unavailable
            logger.info("Falling back to keyword-based grading")
            return self._grade_by_keywords(question, document)
    
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
        """Build grading prompt with improved clarity"""
        prompt = f"""You are a legal document grader. Your task is to determine if a document is relevant to a question.

QUESTION: {question}

DOCUMENT:
{document}
"""
        
        if context:
            prompt += f"\nCONTEXT: {context}\n"
        
        prompt += """
GRADING TASK:
Is the document relevant to answering the question? 

A document is RELEVANT if:
- It contains information that directly answers the question
- It discusses the topic mentioned in the question
- It provides details about what the question asks about

A document is IRRELEVANT if:
- It doesn't contain any information related to the question topic
- It's about a completely different legal topic

RESPOND WITH ONLY THIS FORMAT (no extra text):
{
  "grade": "RELEVANT",
  "score": 0.9,
  "reasoning": "brief explanation"
}

Or:

{
  "grade": "IRRELEVANT", 
  "score": 0.1,
  "reasoning": "brief explanation"
}

Remember: Be generous with RELEVANT. If there is ANY connection between the question and document, mark as RELEVANT."""
        
        return prompt
    
    def _parse_grade_response(
        self,
        response: str,
        question: str,
        document: str
    ) -> Dict[str, Any]:
        """Parse grader response with improved error handling and keyword fallback"""
        import json
        import re
        
        logger.info(f"Raw LLM response (first 500 chars): {response[:500]}")
        
        grade_data = None
        parse_error = None
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    grade_data = json.loads(json_match.group())
                    logger.info(f"Successfully parsed JSON: {grade_data}")
                except json.JSONDecodeError as e:
                    parse_error = str(e)
                    logger.warning(f"JSON parsing error: {e}. Response was: {response[:200]}...")
            else:
                logger.warning("No JSON found in response, attempting keyword parsing")
        except Exception as e:
            parse_error = str(e)
            logger.warning(f"Regex/extraction error: {e}")
        
        # If we couldn't parse JSON, try text-based parsing
        if not grade_data:
            logger.info("Using text-based response parsing")
            grade_data = self._parse_response_by_keywords(response, question, document)
            if parse_error:
                grade_data["reasoning"] = f"Parse error ({parse_error}). Used text parsing: {grade_data.get('reasoning', '')}"
        
        try:
            # Normalize grade
            grade_raw = grade_data.get("grade", "IRRELEVANT").upper() if grade_data else "IRRELEVANT"
            if "RELEVANT" in grade_raw and "PARTIAL" not in grade_raw:
                grade = GradeScore.RELEVANT.value
            elif "PARTIAL" in grade_raw:
                grade = GradeScore.PARTIALLY_RELEVANT.value
            else:
                grade = GradeScore.IRRELEVANT.value
            
            logger.info(f"LLM grade before override: {grade}")
            
            # AGGRESSIVE OVERRIDE: If LLM says irrelevant but ANY keywords match, use keyword-based grading
            if grade == GradeScore.IRRELEVANT.value:
                keyword_match = self._check_keyword_match(question, document)
                logger.info(f"Keyword match ratio: {keyword_match['match_ratio']:.2f}, matched: {keyword_match['matched_keywords']}")
                
                if keyword_match["match_ratio"] > 0.0:  # Changed from 0.5 to 0.0 - any match overrides
                    logger.info(f"OVERRIDING: LLM said IRRELEVANT, but keywords match. Using keyword-based grade.")
                    # Use keyword-based grading instead
                    return self._grade_by_keywords(question, document)
            
            return {
                "grade": grade,
                "score": float(grade_data.get("score", 0.0) if grade_data else 0.0),
                "reasoning": grade_data.get("reasoning", "") if grade_data else "Text-based grading",
                "key_phrases": grade_data.get("key_phrases", []) if grade_data else [],
                "missing_info": grade_data.get("missing_info", "") if grade_data else "",
                "question": question,
                "document_preview": document[:300]
            }
            
        except Exception as e:
            logger.error(f"Error normalizing grade response: {e}")
            # Last resort: keyword matching
            logger.info("Using keyword fallback due to normalization error")
            keyword_match = self._check_keyword_match(question, document)
            if keyword_match["match_ratio"] > 0.0:
                grade = GradeScore.PARTIALLY_RELEVANT.value
            else:
                grade = GradeScore.IRRELEVANT.value
            
            return {
                "grade": grade,
                "score": keyword_match["match_ratio"],
                "reasoning": f"Error fallback, used keyword matching: {', '.join(keyword_match['matched_keywords'])}",
                "question": question,
                "document_preview": document[:300]
            }
    
    def _parse_response_by_keywords(
        self,
        response: str,
        question: str,
        document: str
    ) -> Dict[str, Any]:
        """Parse response by looking for keywords when JSON parsing fails"""
        response_lower = response.lower()
        
        # Look for grade keywords in response
        relevant_count = response_lower.count("relevant")
        partial_count = response_lower.count("partial") + response_lower.count("somewhat") + response_lower.count("partially")
        irrelevant_count = response_lower.count("irrelevant") + response_lower.count("not relevant")
        
        # Determine grade based on keyword presence
        if relevant_count > irrelevant_count and relevant_count > 0 and partial_count == 0:
            grade = "RELEVANT"
            score = 0.8
        elif partial_count > 0 or (relevant_count > 0 and irrelevant_count > 0):
            grade = "PARTIAL"
            score = 0.6
        elif irrelevant_count > relevant_count and irrelevant_count > 0:
            grade = "IRRELEVANT"
            score = 0.2
        else:
            # If we can't determine, default to checking keywords in document
            grade = "IRRELEVANT"
            score = 0.2
        
        # Extract first 100 chars of response as reasoning
        reasoning = response[:100].replace("\n", " ") if response else "Text-based parsing"
        
        return {
            "grade": grade,
            "score": score,
            "reasoning": f"Parsed from response: {reasoning}..."
        }
    
    def _check_keyword_match(
        self,
        question: str,
        document: str
    ) -> Dict[str, Any]:
        """Check if question keywords appear in document with fuzzy matching"""
        import re
        
        # Extract meaningful keywords from question (length > 3)
        question_words = re.findall(r'\b\w{4,}\b', question.lower())
        question_words = [w for w in question_words if len(w) > 3]
        
        # Remove common stop words that aren't meaningful
        stop_words = {'what', 'when', 'where', 'which', 'that', 'this', 'from', 'with', 'have', 'does', 'will', 'would', 'could', 'should'}
        question_words = [w for w in question_words if w not in stop_words]
        
        if not question_words:
            return {"match_ratio": 0.0, "matched_keywords": [], "total_keywords": 0}
        
        doc_lower = document.lower()
        matched = []
        
        # Check for exact and partial matches
        for word in question_words:
            if word in doc_lower:
                matched.append(word)
            else:
                # Check for word stems (e.g., "limit" matches "limitation")
                # Look for the word as a substring or stem
                for stem_len in range(len(word) - 1, max(3, len(word) - 3), -1):
                    stem = word[:stem_len]
                    if stem in doc_lower and len(stem) >= 4:
                        matched.append(f"{word} (matched as '{stem}')")
                        break
        
        match_ratio = len(matched) / len(question_words) if question_words else 0
        
        return {
            "match_ratio": match_ratio,
            "matched_keywords": matched,
            "total_keywords": len(question_words),
            "missing_keywords": [w for w in question_words if w not in matched]
        }
    
    def _grade_by_keywords(
        self,
        question: str,
        document: str
    ) -> Dict[str, Any]:
        """Grade document using simple keyword matching (fallback method)"""
        import re
        
        keyword_match = self._check_keyword_match(question, document)
        match_ratio = keyword_match["match_ratio"]
        matched_keywords = keyword_match["matched_keywords"]
        
        # Determine grade based on keyword match ratio
        if match_ratio >= 0.75:
            grade = GradeScore.RELEVANT.value
            score = 0.8
        elif match_ratio >= 0.5:
            grade = GradeScore.PARTIALLY_RELEVANT.value
            score = 0.6
        elif match_ratio >= 0.25:
            grade = GradeScore.PARTIALLY_RELEVANT.value
            score = 0.4
        else:
            grade = GradeScore.IRRELEVANT.value
            score = 0.2
        
        return {
            "grade": grade,
            "score": score,
            "reasoning": f"Keyword matching ({match_ratio*100:.0f}% match): {', '.join(matched_keywords) if matched_keywords else 'no keywords matched'}",
            "key_phrases": matched_keywords,
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
