"""
Integration Layer for Retrieval System
Connects the retrieval pipeline with the ingestion and API layers
"""
import logging
from typing import List, Dict, Any, Optional

from retriever.retriever import (
    HybridRetriever,
    RetrievedDocument,
    SearchMethod,
    AdvancedFiltering
)
from query_expander import QueryExpander, expand_legal_query
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LegalRetrievalEngine:
    """
    High-level interface for legal document retrieval
    Abstracts away complexity of hybrid search
    """
    
    def __init__(self):
        """Initialize retrieval engine"""
        self.vector_store = VectorStore()
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.query_expander = QueryExpander()
        self.filtering = AdvancedFiltering()
        
        logger.info("Initialized Legal Retrieval Engine")
    
    def answer_legal_question(
        self,
        question: str,
        k: int = 5,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a legal question by searching documents
        
        Args:
            question: Legal question from user
            k: Number of relevant results
            verbose: Return expanded query information
            
        Returns:
            Dict with results, explanations, and metadata
        """
        logger.info(f"Answering question: {question}")
        
        # Expand query
        expanded = self.query_expander.expand_query(question)
        
        # Retrieve documents
        results = self.hybrid_retriever.retrieve(
            query=question,
            k=k,
            method=SearchMethod.HYBRID,
            query_expansion=True,
            rerank=True
        )
        
        # Format response
        response = {
            "original_question": question,
            "results": [doc.to_dict() for doc in results],
            "num_results": len(results),
            "retrieval_method": "hybrid"
        }
        
        if verbose:
            response["query_expansion"] = {
                "expanded_query": expanded["expanded_query"],
                "query_type": expanded["query_type"],
                "search_terms": expanded["search_terms"][:5],
                "contexts": expanded["contexts"]
            }
        
        return response
    
    def search_with_method(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 5,
        expand_query: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search using specific method
        
        Args:
            query: Search query
            method: "semantic", "keyword", or "hybrid"
            k: Number of results
            expand_query: Whether to expand query
            
        Returns:
            List of result dictionaries
        """
        search_method = SearchMethod[method.upper()]
        
        results = self.hybrid_retriever.retrieve(
            query=query,
            k=k,
            method=search_method,
            query_expansion=expand_query,
            rerank=True
        )
        
        return [doc.to_dict() for doc in results]
    
    def search_by_category(
        self,
        category: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Search for documents in a legal category
        
        Args:
            category: Legal category (e.g., "liability", "payment", "confidentiality")
            k: Number of results
            
        Returns:
            Category search results
        """
        category_queries = {
            "liability": "What are the limitations on liability and indemnification obligations?",
            "payment": "What are the payment terms, due dates, and late payment penalties?",
            "confidentiality": "What information must be kept confidential?",
            "termination": "How can this agreement be terminated and what are the consequences?",
            "ip": "What intellectual property rights are protected?",
            "warranties": "What warranties and representations are made?",
            "jurisdiction": "What jurisdiction and governing law apply?",
            "obligations": "What are the primary obligations of each party?",
        }
        
        if category.lower() not in category_queries:
            logger.warning(f"Unknown category: {category}")
            return {"error": f"Unknown category: {category}"}
        
        query = category_queries[category.lower()]
        results = self.answer_legal_question(query, k=k)
        
        results["category"] = category
        return results
    
    def compare_clauses(
        self,
        clause1_query: str,
        clause2_query: str,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare two different clauses/concepts
        
        Args:
            clause1_query: Query for first clause
            clause2_query: Query for second clause
            k: Number of results per clause
            
        Returns:
            Comparison of results
        """
        results1 = self.hybrid_retriever.retrieve(clause1_query, k=k)
        results2 = self.hybrid_retriever.retrieve(clause2_query, k=k)
        
        return {
            "clause1": {
                "query": clause1_query,
                "results": [doc.to_dict() for doc in results1]
            },
            "clause2": {
                "query": clause2_query,
                "results": [doc.to_dict() for doc in results2]
            }
        }
    
    def find_related_clauses(
        self,
        clause_content: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find clauses related to a given clause
        
        Args:
            clause_content: Content of the clause
            k: Number of related clauses
            
        Returns:
            List of related clauses
        """
        results = self.hybrid_retriever.retrieve(
            query=clause_content,
            k=k,
            method=SearchMethod.SEMANTIC,  # Use semantic for finding similar content
            query_expansion=False,  # Don't expand for related search
            rerank=True
        )
        
        return [doc.to_dict() for doc in results]
    
    def advanced_search(
        self,
        query: str,
        k: int = 5,
        section_title: Optional[str] = None,
        chunk_type: Optional[str] = None,
        document_id: Optional[str] = None,
        relevance_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple filters
        
        Args:
            query: Search query
            k: Number of results
            section_title: Optional section filter
            chunk_type: Optional chunk type filter
            document_id: Optional document filter
            relevance_threshold: Minimum relevance score
            
        Returns:
            Filtered search results
        """
        # Retrieve results
        results = self.hybrid_retriever.retrieve(
            query=query,
            k=k*2,  # Get extra results for filtering
            method=SearchMethod.HYBRID,
            query_expansion=True,
            rerank=True
        )
        
        # Apply filters
        if section_title:
            results = self.filtering.filter_by_section(results, section_title)
        
        if chunk_type:
            results = self.filtering.filter_by_chunk_type(results, chunk_type)
        
        if document_id:
            results = self.filtering.filter_by_document(results, document_id)
        
        results = self.filtering.filter_by_relevance_threshold(results, relevance_threshold)
        
        # Apply diversity filter to get varied results
        results = self.filtering.filter_diversity(results, max_per_section=1)
        
        return [doc.to_dict() for doc in results[:k]]
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get suggestions for completing a query
        
        Args:
            partial_query: Partial user query
            
        Returns:
            List of suggested completions
        """
        # Use query expander to get context
        expanded = self.query_expander.expand_query(partial_query)
        
        suggestions = []
        
        # Add expanded terms as suggestions
        suggestions.extend(expanded["search_terms"][:5])
        
        # Add legal contexts
        suggestions.extend(expanded["contexts"][:3])
        
        return list(dict.fromkeys(suggestions))[:10]  # Remove duplicates and limit


class RetrievalMetrics:
    """Track and analyze retrieval performance"""
    
    def __init__(self):
        self.queries = []
        self.results = []
    
    def log_query(self, query: str, results: List[RetrievedDocument], method: str):
        """Log a query and its results"""
        self.queries.append({
            "query": query,
            "method": method,
            "num_results": len(results),
            "avg_score": sum(r.final_score for r in results) / len(results) if results else 0,
            "top_score": results[0].final_score if results else 0
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        if not self.queries:
            return {}
        
        return {
            "total_queries": len(self.queries),
            "avg_results_per_query": sum(q["num_results"] for q in self.queries) / len(self.queries),
            "avg_top_score": sum(q["top_score"] for q in self.queries) / len(self.queries),
            "method_distribution": self._get_method_distribution()
        }
    
    def _get_method_distribution(self) -> Dict[str, int]:
        """Get distribution of search methods used"""
        distribution = {}
        for query in self.queries:
            method = query["method"]
            distribution[method] = distribution.get(method, 0) + 1
        return distribution


# Convenience functions
def ask_legal_question(
    question: str,
    k: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to ask a legal question
    """
    engine = LegalRetrievalEngine()
    return engine.answer_legal_question(question, k=k, verbose=verbose)


def legal_search(
    query: str,
    method: str = "hybrid",
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for legal search
    """
    engine = LegalRetrievalEngine()
    return engine.search_with_method(query, method=method, k=k)


def search_legal_category(category: str, k: int = 5) -> Dict[str, Any]:
    """
    Convenience function for category search
    """
    engine = LegalRetrievalEngine()
    return engine.search_by_category(category, k=k)
