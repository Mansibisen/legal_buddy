"""
Query Expansion for Legal Document Retrieval
Transforms simple user queries into comprehensive legal searches
"""
import logging
import re
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from config import settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of legal queries"""
    PAYMENT = "payment"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    IP = "intellectual_property"
    OBLIGATIONS = "obligations"
    WARRANTIES = "warranties"
    JURISDICTION = "jurisdiction"
    GENERAL = "general"


class LegalOntology:
    """Legal domain knowledge for query expansion"""
    
    # Legal terminology expansions
    SYNONYMS = {
        "payment": [
            "payment", "compensation", "fee", "charge", "cost", "price",
            "amount", "invoice", "bill", "remittance", "transfer", "money",
            "consideration", "premium", "rate", "tariff"
        ],
        "late": [
            "late", "overdue", "delay", "delinquent", "tardy", "arrears",
            "past due", "failure to pay", "non-payment"
        ],
        "penalty": [
            "penalty", "fine", "sanction", "charge", "fee", "surcharge",
            "assessment", "consequence", "damages", "liability"
        ],
        "interest": [
            "interest", "rate", "APR", "annual percentage rate", "accrual",
            "finance charge", "cost of money", "time value"
        ],
        "grace": [
            "grace", "grace period", "deferment", "forbearance", "extension",
            "moratorium", "reprieve", "leniency"
        ],
        "breach": [
            "breach", "violation", "infringement", "default", "non-compliance",
            "failure", "transgression", "rupture"
        ],
        "liable": [
            "liable", "responsibility", "obligation", "accountable", "bound",
            "answerable", "liable", "subject to"
        ],
        "damages": [
            "damages", "compensation", "loss", "injury", "harm", "injury",
            "reparation", "restitution", "recovery"
        ],
        "indemnify": [
            "indemnify", "indemnification", "hold harmless", "defend",
            "protect", "reimburse", "save harmless", "guarantee"
        ],
        "confidential": [
            "confidential", "secret", "private", "proprietary", "classified",
            "sensitive", "restricted", "privileged", "non-disclosure"
        ],
        "terminate": [
            "terminate", "termination", "cancel", "cancellation", "end",
            "cessation", "dissolve", "revoke", "discontinue"
        ],
        "intellectual property": [
            "intellectual property", "copyright", "patent", "trademark",
            "trade secret", "ownership", "invention", "creation"
        ],
        "warranty": [
            "warranty", "warrant", "guarantee", "representation", "assurance",
            "promise", "condition", "covenant"
        ],
        "jurisdiction": [
            "jurisdiction", "governing law", "venue", "forum", "court",
            "applicable law", "legal authority"
        ],
    }
    
    # Common legal phrases and their expansions
    PHRASE_PATTERNS = {
        r"(?i)what.*late.*fee": [
            "late payment", "late fees", "payment penalties",
            "interest charges", "grace periods", "delinquency"
        ],
        r"(?i)what.*pay.*deadline": [
            "payment terms", "due date", "payment schedule",
            "invoice terms", "payment deadline"
        ],
        r"(?i)what.*break.*contract": [
            "breach of contract", "default", "violation",
            "damages for breach", "remedies", "consequences"
        ],
        r"(?i)how.*terminate": [
            "termination clause", "termination conditions",
            "termination rights", "termination process"
        ],
        r"(?i)what.*keep.*secret": [
            "confidentiality", "non-disclosure", "proprietary",
            "confidential information", "trade secrets"
        ],
        r"(?i)who.*responsible": [
            "liability", "responsibility", "indemnification",
            "hold harmless", "damages"
        ],
        r"(?i)what.*use.*name": [
            "intellectual property", "trademark", "copyright",
            "ownership rights", "use rights"
        ],
    }
    
    # Query expansion rules by type
    EXPANSION_RULES = {
        QueryType.PAYMENT: {
            "primary_terms": ["payment", "fee", "cost", "invoice", "billing"],
            "related_terms": ["terms", "schedule", "due date", "late", "penalty"],
            "legal_contexts": [
                "payment terms and conditions",
                "payment obligations and deadlines",
                "late payment penalties and interest",
                "payment methods and procedures"
            ]
        },
        QueryType.LIABILITY: {
            "primary_terms": ["liability", "damages", "responsible", "fault"],
            "related_terms": ["indemnify", "hold harmless", "breach", "limit"],
            "legal_contexts": [
                "liability limitations and caps",
                "liability allocation and responsibility",
                "indemnification and hold harmless",
                "damages and remedies"
            ]
        },
        QueryType.CONFIDENTIALITY: {
            "primary_terms": ["confidential", "secret", "proprietary", "disclosure"],
            "related_terms": ["protected", "restricted", "access", "unauthorized"],
            "legal_contexts": [
                "confidentiality obligations",
                "non-disclosure agreements",
                "information protection requirements",
                "proprietary information safeguards"
            ]
        },
        QueryType.TERMINATION: {
            "primary_terms": ["terminate", "end", "cancel", "discontinue"],
            "related_terms": ["notice", "effective", "wind down", "transition"],
            "legal_contexts": [
                "termination rights and conditions",
                "termination notice requirements",
                "termination procedures and obligations",
                "post-termination obligations"
            ]
        },
        QueryType.WARRANTIES: {
            "primary_terms": ["warranty", "guarantee", "represent", "warrant"],
            "related_terms": ["condition", "standard", "performance", "quality"],
            "legal_contexts": [
                "warranty representations",
                "warranty disclaimers and limitations",
                "warranty breach remedies",
                "warranty conditions and standards"
            ]
        },
    }


class QueryExpander:
    """
    Expands simple user queries into comprehensive legal searches
    """
    
    def __init__(self):
        self.ontology = LegalOntology()
        self.max_expansion_terms = 15
        
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand user query with legal terminology and context
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict with:
            - original_query: Original user query
            - expanded_query: Expanded search query
            - query_type: Classified query type
            - search_terms: List of individual search terms
            - contexts: Legal contexts to search for
            - weights: Term importance weights
        """
        logger.info(f"Expanding query: {query}")
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Extract core terms
        core_terms = self._extract_core_terms(query)
        
        # Expand with synonyms and related terms
        expanded_terms = self._expand_with_synonyms(core_terms)
        
        # Get legal contexts
        contexts = self._get_legal_contexts(query_type, core_terms)
        
        # Build expanded query
        expanded_query = self._build_expanded_query(expanded_terms, contexts)
        
        # Calculate term weights
        weights = self._calculate_term_weights(core_terms, expanded_terms)
        
        result = {
            "original_query": query,
            "expanded_query": expanded_query,
            "query_type": query_type.value,
            "search_terms": expanded_terms[:self.max_expansion_terms],
            "core_terms": core_terms,
            "contexts": contexts,
            "weights": weights,
            "expansion_ratio": len(expanded_terms) / max(len(core_terms), 1)
        }
        
        logger.info(f"Expanded to {len(expanded_terms)} terms")
        return result
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of legal query"""
        query_lower = query.lower()
        
        # Payment queries
        if any(word in query_lower for word in ["payment", "fee", "pay", "cost", "price", "bill"]):
            return QueryType.PAYMENT
        
        # Liability queries
        if any(word in query_lower for word in ["liable", "liability", "responsible", "damage", "indemnif"]):
            return QueryType.LIABILITY
        
        # Confidentiality queries
        if any(word in query_lower for word in ["confidential", "secret", "private", "disclosure", "proprietary"]):
            return QueryType.CONFIDENTIALITY
        
        # Termination queries
        if any(word in query_lower for word in ["terminate", "cancel", "end", "discontin"]):
            return QueryType.TERMINATION
        
        # IP queries
        if any(word in query_lower for word in ["intellectual property", "patent", "copyright", "trademark"]):
            return QueryType.IP
        
        # Obligation queries
        if any(word in query_lower for word in ["obligation", "must", "shall", "required", "responsible"]):
            return QueryType.OBLIGATIONS
        
        # Warranty queries
        if any(word in query_lower for word in ["warrant", "guarantee", "condition", "representation"]):
            return QueryType.WARRANTIES
        
        # Jurisdiction queries
        if any(word in query_lower for word in ["jurisdiction", "governing law", "venue", "court"]):
            return QueryType.JURISDICTION
        
        return QueryType.GENERAL
    
    def _extract_core_terms(self, query: str) -> List[str]:
        """Extract core terms from query"""
        # Remove common stop words
        stop_words = {
            "what", "is", "the", "a", "an", "and", "or", "but", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "as", "be", "are",
            "can", "could", "would", "should", "do", "does", "did", "have"
        }
        
        # Split query into terms
        terms = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and get core terms
        core_terms = [t for t in terms if t not in stop_words and len(t) > 2]
        
        return core_terms
    
    def _expand_with_synonyms(self, core_terms: List[str]) -> List[str]:
        """Expand terms with synonyms"""
        expanded = set(core_terms)
        
        for term in core_terms:
            # Check direct matches
            if term in self.ontology.SYNONYMS:
                expanded.update(self.ontology.SYNONYMS[term])
            
            # Check partial matches
            for key, synonyms in self.ontology.SYNONYMS.items():
                if term in key or key in term:
                    expanded.update(synonyms)
        
        return list(expanded)
    
    def _get_legal_contexts(self, query_type: QueryType, core_terms: List[str]) -> List[str]:
        """Get relevant legal contexts for the query"""
        contexts = []
        
        # Get contexts from query type rules
        if query_type in self.ontology.EXPANSION_RULES:
            rules = self.ontology.EXPANSION_RULES[query_type]
            contexts.extend(rules.get("legal_contexts", []))
        
        # Add context phrases based on core terms
        for term in core_terms:
            # Check phrase patterns
            for pattern, phrases in self.ontology.PHRASE_PATTERNS.items():
                if re.search(pattern, term):
                    contexts.extend(phrases)
        
        return list(set(contexts))[:10]  # Limit to top 10 contexts
    
    def _build_expanded_query(self, expanded_terms: List[str], contexts: List[str]) -> str:
        """Build expanded search query string"""
        # Combine expanded terms and contexts
        all_items = expanded_terms + contexts
        
        # Remove duplicates and take top items
        unique_items = list(set(all_items))[:10]
        
        # Join with OR for broader search
        expanded_query = " OR ".join(unique_items)
        
        return expanded_query
    
    def _calculate_term_weights(self, core_terms: List[str], expanded_terms: List[str]) -> Dict[str, float]:
        """Calculate importance weights for terms"""
        weights = {}
        
        # Core terms get highest weight
        for term in core_terms:
            weights[term] = 1.0
        
        # Expanded terms get lower weights based on type of relationship
        for term in expanded_terms:
            if term not in weights:
                # Check if synonym
                for key, synonyms in self.ontology.SYNONYMS.items():
                    if term in synonyms and key in core_terms:
                        weights[term] = 0.8
                        break
                else:
                    weights[term] = 0.5
        
        return weights


class QueryOptimizer:
    """Optimizes queries for different search backends"""
    
    @staticmethod
    def optimize_for_semantic(expanded_query: Dict[str, Any]) -> str:
        """Optimize expanded query for semantic/vector search"""
        # For semantic search, use the expanded query directly
        # Focus on meaning rather than exact terms
        terms = expanded_query["search_terms"][:5]  # Top 5 terms
        return " ".join(terms)
    
    @staticmethod
    def optimize_for_keyword(expanded_query: Dict[str, Any]) -> List[str]:
        """Optimize expanded query for keyword/BM25 search"""
        # For keyword search, use core terms with high weight
        core_terms = expanded_query["core_terms"]
        # Add related synonyms
        related = expanded_query["search_terms"][5:10]
        return core_terms + related
    
    @staticmethod
    def optimize_for_structured(expanded_query: Dict[str, Any]) -> Dict[str, List[str]]:
        """Optimize query for structured field search"""
        return {
            "primary_terms": expanded_query["core_terms"],
            "related_terms": expanded_query["search_terms"][5:10],
            "contexts": expanded_query["contexts"]
        }


class QueryDecomposer:
    """Breaks down complex queries into simpler sub-queries"""
    
    @staticmethod
    def decompose(query: str) -> List[str]:
        """
        Break down complex query into simpler sub-queries
        
        Args:
            query: Complex query with multiple clauses
            
        Returns:
            List of simpler queries
        """
        queries = []
        
        # Split on 'and' conjunctions
        and_parts = re.split(r'\band\b', query, flags=re.IGNORECASE)
        
        for part in and_parts:
            # Split on common disjunctions
            or_parts = re.split(r'\bor\b', part, flags=re.IGNORECASE)
            queries.extend([p.strip() for p in or_parts if p.strip()])
        
        return queries


def expand_legal_query(query: str, strategy: str = "standard") -> Dict[str, Any]:
    """
    Convenience function to expand a legal query
    
    Args:
        query: User query
        strategy: "standard", "aggressive", or "conservative"
        
    Returns:
        Expanded query information
    """
    expander = QueryExpander()
    result = expander.expand_query(query)
    
    # Adjust based on strategy
    if strategy == "aggressive":
        # Keep more expansion terms
        result["search_terms"] = result["search_terms"][:20]
        
    elif strategy == "conservative":
        # Keep only core terms and close synonyms
        result["search_terms"] = result["search_terms"][:5]
    
    return result
