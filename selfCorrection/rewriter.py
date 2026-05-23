"""
Query Rewriter for Self-Correction Loop
Improves and rewrites failed queries based on feedback
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from langchain_ollama import OllamaLLM
from config import settings

logger = logging.getLogger(__name__)


class RewriteStrategy(Enum):
    """Strategies for query rewriting"""
    EXPAND = "expand"          # Add more specific terms
    SIMPLIFY = "simplify"      # Simplify overly complex queries
    REPHRASE = "rephrase"      # Rephrase in different way
    FOCUS = "focus"            # Focus on core question
    DECOMPOSE = "decompose"    # Break into sub-queries


class QueryRewriter:
    """
    Rewrites and improves queries that failed to retrieve relevant documents
    """
    
    def __init__(self, model: str = None):
        """
        Initialize query rewriter
        
        Args:
            model: LLM model to use for rewriting
        """
        self.model = model or settings.OLLAMA_MODEL
        self.client = OllamaLLM(
            model=self.model,
            base_url=settings.OLLAMA_BASE_URL
        )
    
    def rewrite_query(
        self,
        original_query: str,
        failure_reason: str,
        context: Optional[str] = None,
        strategy: str = "expand"
    ) -> Dict[str, Any]:
        """
        Rewrite a query that failed to return relevant documents
        
        Args:
            original_query: Original query that failed
            failure_reason: Why the original query failed
            context: Optional context about the document/domain
            strategy: Rewriting strategy
            
        Returns:
            Rewritten query with explanation
        """
        logger.info(f"Rewriting query: {original_query}")
        logger.info(f"Failure reason: {failure_reason}")
        
        prompt = self._build_rewrite_prompt(
            original_query,
            failure_reason,
            context,
            strategy
        )
        
        try:
            response_text = self.client.invoke(prompt)
            
            # Parse response
            result = self._parse_rewrite_response(response_text, original_query)
            
            return result
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return {
                "original_query": original_query,
                "rewritten_query": self._fallback_rewrite(original_query),
                "strategy_used": strategy,
                "explanation": f"Rewriting error: {str(e)}",
                "alternative_queries": []
            }
    
    def rewrite_with_multiple_strategies(
        self,
        original_query: str,
        failure_reason: str,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple rewritten queries using different strategies
        
        Args:
            original_query: Original query
            failure_reason: Why it failed
            context: Optional context
            
        Returns:
            List of rewritten queries with different strategies
        """
        strategies = [
            RewriteStrategy.EXPAND.value,
            RewriteStrategy.REPHRASE.value,
            RewriteStrategy.FOCUS.value
        ]
        
        results = []
        for strategy in strategies:
            result = self.rewrite_query(
                original_query,
                failure_reason,
                context,
                strategy
            )
            results.append(result)
        
        return results
    
    def _build_rewrite_prompt(
        self,
        original_query: str,
        failure_reason: str,
        context: Optional[str],
        strategy: str
    ) -> str:
        """Build query rewriting prompt"""
        prompt = f"""You are an expert at reformulating search queries for legal document retrieval.

ORIGINAL QUERY: {original_query}

WHY IT FAILED: {failure_reason}

STRATEGY: {strategy}
"""
        
        if context:
            prompt += f"\nDOCUMENT CONTEXT: {context}\n"
        
        prompt += f"""
Based on the failure reason, rewrite the query using the {strategy} strategy:

STRATEGY GUIDELINES:
- EXPAND: Add more specific legal terms, synonyms, and related concepts
- SIMPLIFY: Remove jargon, make the query more straightforward
- REPHRASE: Express the same idea in a different way
- FOCUS: Narrow down to the core question, remove ambiguity
- DECOMPOSE: Break into multiple specific sub-queries

Respond in JSON format:
{{
  "rewritten_query": "<new query string>",
  "explanation": "<why this rewrite should work>",
  "confidence": <0.0-1.0>,
  "alternative_queries": ["<alternative 1>", "<alternative 2>"],
  "key_improvements": ["<improvement 1>", "<improvement 2>"]
}}
"""
        return prompt
    
    def _parse_rewrite_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse rewriter response"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                rewrite_data = json.loads(json_match.group())
            else:
                rewrite_data = {}
            
            return {
                "original_query": original_query,
                "rewritten_query": rewrite_data.get("rewritten_query", original_query),
                "explanation": rewrite_data.get("explanation", ""),
                "confidence": float(rewrite_data.get("confidence", 0.5)),
                "alternative_queries": rewrite_data.get("alternative_queries", []),
                "key_improvements": rewrite_data.get("key_improvements", [])
            }
            
        except Exception as e:
            logger.error(f"Error parsing rewrite response: {e}")
            return {
                "original_query": original_query,
                "rewritten_query": self._fallback_rewrite(original_query),
                "explanation": f"Parse error: {str(e)}",
                "confidence": 0.3,
                "alternative_queries": [],
                "key_improvements": []
            }
    
    def _fallback_rewrite(self, query: str) -> str:
        """Fallback query rewriting if AI fails"""
        import re
        
        # Try simple improvements
        rewritten = query
        
        # Add legal terminology if missing
        legal_terms = {
            "payment": ["fee", "cost", "amount", "consideration"],
            "break": ["breach", "violation", "default"],
            "money": ["payment", "compensation", "consideration"],
            "rules": ["terms", "conditions", "obligations", "provisions"]
        }
        
        for term, alternatives in legal_terms.items():
            if term in rewritten.lower():
                rewritten = rewritten + " " + " OR ".join(alternatives)
                break
        
        return rewritten or query


class QueryImprover:
    """Systematic query improvement without AI"""
    
    @staticmethod
    def add_synonyms(query: str) -> str:
        """Add legal synonyms to query"""
        synonyms = {
            "pay": "pay OR payment OR invoice OR bill OR remittance",
            "late": "late OR overdue OR delinquent OR delay OR tardy",
            "break": "break OR breach OR violate OR default",
            "money": "money OR payment OR funds OR amount OR cost",
            "responsible": "responsible OR liable OR accountable OR responsible",
            "secret": "secret OR confidential OR proprietary OR private",
            "end": "end OR terminate OR cancel OR discontinue"
        }
        
        improved = query
        for term, replacement in synonyms.items():
            if term in query.lower():
                improved = improved.replace(term, replacement)
        
        return improved
    
    @staticmethod
    def add_legal_context(query: str) -> str:
        """Add legal context terms to query"""
        # Add common legal clauses if not present
        legal_contexts = [
            "clause",
            "section",
            "provision",
            "article",
            "paragraph"
        ]
        
        if not any(term in query.lower() for term in legal_contexts):
            query = query + " clause"
        
        return query
    
    @staticmethod
    def make_more_specific(query: str) -> str:
        """Make query more specific"""
        specificity_terms = {
            "important": "critical OR material OR substantial OR significant",
            "what": "what are the OR what is the OR describe the",
            "how": "how does OR how do OR explain",
            "why": "why OR reason OR basis"
        }
        
        improved = query
        for term, replacement in specificity_terms.items():
            if term in query.lower():
                improved = improved.replace(term, replacement)
        
        return improved


class RetryStrategy:
    """Strategy for retrying failed queries"""
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize retry strategy
        
        Args:
            max_retries: Maximum number of retries
        """
        self.max_retries = max_retries
        self.rewriter = QueryRewriter()
    
    def should_retry(
        self,
        attempt: int,
        grade_assessment: Dict[str, Any]
    ) -> bool:
        """
        Determine if query should be retried
        
        Args:
            attempt: Current attempt number
            grade_assessment: Assessment from document grader
            
        Returns:
            Whether to retry
        """
        if attempt >= self.max_retries:
            logger.info(f"Max retries ({self.max_retries}) reached")
            return False
        
        # Retry if no relevant documents found
        if grade_assessment.get("relevant_count", 0) == 0:
            return True
        
        # Retry if very low relevance ratio
        if grade_assessment.get("relevance_ratio", 0) < 0.3:
            return True
        
        return False
    
    def get_retry_feedback(
        self,
        grade_assessment: Dict[str, Any]
    ) -> str:
        """
        Generate feedback for query rewriting
        
        Args:
            grade_assessment: Assessment from grader
            
        Returns:
            Feedback string explaining why retry is needed
        """
        relevant = grade_assessment.get("relevant_count", 0)
        partial = grade_assessment.get("partial_count", 0)
        total = grade_assessment.get("total_documents", 0)
        
        if relevant == 0 and partial == 0:
            return "No documents found that address the question. Try using different terminology or broader search terms."
        
        if relevant == 0 and partial > 0:
            return "Found partially relevant documents but none fully answer the question. Try being more specific or adding clarifying details."
        
        return "Relevance score too low. Try rephrasing the question or using more specific legal terminology."


def rewrite_failed_query(
    original_query: str,
    failure_reason: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to rewrite a failed query
    
    Args:
        original_query: Original query that failed
        failure_reason: Why it failed
        context: Optional context
        
    Returns:
        Rewritten query information
    """
    rewriter = QueryRewriter()
    return rewriter.rewrite_query(original_query, failure_reason, context)
