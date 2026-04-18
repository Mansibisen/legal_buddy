"""
Hybrid Retrieval System
Combines Semantic Search (Vector Similarity) with Keyword Search (BM25)
for comprehensive legal document retrieval
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import Counter
import re

from langchain_core.documents import Document

from ingestion.vector_store import VectorStore
from query_expander import QueryExpander, QueryOptimizer

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Search method types"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with scores"""
    content: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "final_score": round(self.final_score, 4),
            "rank": self.rank
        }


class BM25Retriever:
    """
    BM25 (Best Match 25) keyword search implementation
    Probabilistic ranking function for information retrieval
    """
    
    def __init__(self, documents: List[Dict[str, Any]] = None):
        """
        Initialize BM25 retriever
        
        Args:
            documents: List of documents to build index from
        """
        self.documents = documents or []
        self.doc_index = {}
        self.idf_cache = {}
        self.k1 = 1.5  # Term frequency saturation point
        self.b = 0.75  # Length normalization
        self.avg_doc_length = 0
        
        if documents:
            self._build_index(documents)
    
    def _build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents"""
        self.documents = documents
        doc_frequency = Counter()
        total_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            
            self.doc_index[i] = {
                "tokens": tokens,
                "length": len(tokens),
                "content": content
            }
            
            # Track document frequency
            for token in set(tokens):
                doc_frequency[token] += 1
            
            total_length += len(tokens)
        
        # Calculate average document length
        self.avg_doc_length = total_length / len(documents) if documents else 0
        
        # Calculate IDF (Inverse Document Frequency)
        total_docs = len(documents)
        for token, freq in doc_frequency.items():
            self.idf_cache[token] = math.log(
                (total_docs - freq + 0.5) / (freq + 0.5) + 1
            )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search documents using BM25
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = {}
        
        for doc_id, doc_data in self.doc_index.items():
            score = 0.0
            doc_tokens = doc_data["tokens"]
            doc_length = doc_data["length"]
            
            # Calculate BM25 score
            for token in query_tokens:
                # Term frequency in document
                tf = doc_tokens.count(token)
                
                # Get IDF
                idf = self.idf_cache.get(token, 0)
                
                # BM25 formula
                normalized_length = doc_length / self.avg_doc_length if self.avg_doc_length > 0 else 1
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * normalized_length)
                
                score += idf * (numerator / denominator)
            
            scores[doc_id] = score
        
        # Sort by score and return top k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [
            (self.doc_index[doc_id], score)
            for doc_id, score in ranked
        ]


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: VectorStore instance for semantic search
        """
        self.vector_store = vector_store or VectorStore()
        self.query_expander = QueryExpander()
        self.query_optimizer = QueryOptimizer()
        self.bm25_retriever = BM25Retriever()
        self.documents_cache = []
        
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
    
    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """
        Set search method weights
        
        Args:
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        
        logger.info(f"Weights - Semantic: {self.semantic_weight:.2f}, Keyword: {self.keyword_weight:.2f}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: SearchMethod = SearchMethod.HYBRID,
        query_expansion: bool = True,
        rerank: bool = True
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using specified method
        
        Args:
            query: User query
            k: Number of results
            method: Search method (semantic, keyword, or hybrid)
            query_expansion: Whether to expand query
            rerank: Whether to rerank results
            
        Returns:
            List of RetrievedDocument objects
        """
        logger.info(f"Retrieving with {method.value} search: {query}")
        
        # Expand query if requested
        if query_expansion:
            expanded = self.query_expander.expand_query(query)
            logger.info(f"Query expanded to: {expanded['expanded_query']}")
        else:
            expanded = None
        
        # Perform search based on method
        if method == SearchMethod.SEMANTIC:
            results = self._semantic_search(query, expanded, k)
            
        elif method == SearchMethod.KEYWORD:
            results = self._keyword_search(query, expanded, k)
            
        else:  # HYBRID
            results = self._hybrid_search(query, expanded, k)
        
        # Rerank results if requested
        if rerank and len(results) > 1:
            results = self._rerank_results(results, query, expanded)
        
        # Assign final ranks
        for i, doc in enumerate(results, 1):
            doc.rank = i
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        expanded: Optional[Dict[str, Any]],
        k: int
    ) -> List[RetrievedDocument]:
        """Perform semantic search"""
        # Use expanded query for semantic search
        search_query = query
        if expanded:
            search_query = self.query_optimizer.optimize_for_semantic(expanded)
        
        # Search vector store
        results = self.vector_store.search(search_query, k=k*2)  # Get more for filtering
        
        retrieved = []
        for result in results[:k]:
            doc = RetrievedDocument(
                content=result["content"],
                metadata=result["metadata"],
                semantic_score=result["relevance_score"]
            )
            retrieved.append(doc)
        
        return retrieved
    
    def _keyword_search(
        self,
        query: str,
        expanded: Optional[Dict[str, Any]],
        k: int
    ) -> List[RetrievedDocument]:
        """Perform keyword search"""
        # Get keyword search query
        if expanded:
            keywords = self.query_optimizer.optimize_for_keyword(expanded)
            search_query = " ".join(keywords)
        else:
            search_query = query
        
        # Search with BM25
        results = self.bm25_retriever.search(search_query, k=k*2)
        
        retrieved = []
        for doc_data, score in results[:k]:
            doc = RetrievedDocument(
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                keyword_score=score
            )
            retrieved.append(doc)
        
        return retrieved
    
    def _hybrid_search(
        self,
        query: str,
        expanded: Optional[Dict[str, Any]],
        k: int
    ) -> List[RetrievedDocument]:
        """Perform hybrid search"""
        # Get more results from each method
        k_expanded = int(k * 1.5)
        
        # Semantic search
        semantic_results = self._semantic_search(query, expanded, k_expanded)
        
        # Keyword search
        keyword_results = self._keyword_search(query, expanded, k_expanded)
        
        # Merge and deduplicate results
        merged = self._merge_results(semantic_results, keyword_results, k)
        
        return merged
    
    def _merge_results(
        self,
        semantic_results: List[RetrievedDocument],
        keyword_results: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """Merge and score results from both methods"""
        # Create a map of content to documents
        doc_map = {}
        
        # Add semantic results
        for doc in semantic_results:
            key = self._create_doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            else:
                # Update with semantic score
                doc_map[key].semantic_score = max(doc_map[key].semantic_score, doc.semantic_score)
        
        # Add keyword results
        for doc in keyword_results:
            key = self._create_doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            else:
                # Update with keyword score
                doc_map[key].keyword_score = max(doc_map[key].keyword_score, doc.keyword_score)
        
        # Calculate final scores
        merged = []
        for doc in doc_map.values():
            # Normalize scores to 0-1 range
            semantic_score = min(doc.semantic_score, 1.0)
            keyword_score = min(doc.keyword_score, 1.0) if doc.keyword_score > 0 else 0
            
            # Combine scores
            final_score = (
                semantic_score * self.semantic_weight +
                keyword_score * self.keyword_weight
            )
            
            doc.final_score = final_score
            merged.append(doc)
        
        # Sort by final score
        merged.sort(key=lambda x: x.final_score, reverse=True)
        
        return merged[:k]
    
    def _create_doc_key(self, doc: RetrievedDocument) -> str:
        """Create a unique key for document deduplication"""
        # Use first 100 chars of content as key
        return doc.content[:100]
    
    def _rerank_results(
        self,
        results: List[RetrievedDocument],
        query: str,
        expanded: Optional[Dict[str, Any]]
    ) -> List[RetrievedDocument]:
        """Re-rank results using additional signals"""
        # Calculate relevance scores based on various factors
        for doc in results:
            relevance_score = self._calculate_relevance(doc, query, expanded)
            
            # Boost score based on relevance
            doc.final_score = doc.final_score * (1 + 0.1 * relevance_score)
        
        # Re-sort
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def _calculate_relevance(
        self,
        doc: RetrievedDocument,
        query: str,
        expanded: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate relevance score for document"""
        score = 0.0
        
        # Check if document contains exact query terms
        if query.lower() in doc.content.lower():
            score += 0.5
        
        # Check for query terms in section title
        section_title = doc.metadata.get("section_title", "").lower()
        if query.lower() in section_title:
            score += 0.3
        
        # Check for expanded terms
        if expanded:
            core_terms = expanded.get("core_terms", [])
            content_lower = doc.content.lower()
            
            for term in core_terms:
                if term.lower() in content_lower:
                    score += 0.1 / len(core_terms) if core_terms else 0
        
        return min(score, 1.0)  # Cap at 1.0


class AdvancedFiltering:
    """Advanced result filtering options"""
    
    @staticmethod
    def filter_by_section(
        results: List[RetrievedDocument],
        section_title: str
    ) -> List[RetrievedDocument]:
        """Filter results by section"""
        return [
            doc for doc in results
            if section_title.lower() in doc.metadata.get("section_title", "").lower()
        ]
    
    @staticmethod
    def filter_by_chunk_type(
        results: List[RetrievedDocument],
        chunk_type: str
    ) -> List[RetrievedDocument]:
        """Filter results by chunk type"""
        return [
            doc for doc in results
            if doc.metadata.get("chunk_type") == chunk_type
        ]
    
    @staticmethod
    def filter_by_document(
        results: List[RetrievedDocument],
        document_id: str
    ) -> List[RetrievedDocument]:
        """Filter results from specific document"""
        return [
            doc for doc in results
            if doc.metadata.get("document_id") == document_id
        ]
    
    @staticmethod
    def filter_by_relevance_threshold(
        results: List[RetrievedDocument],
        threshold: float
    ) -> List[RetrievedDocument]:
        """Filter by relevance score threshold"""
        return [doc for doc in results if doc.final_score >= threshold]
    
    @staticmethod
    def filter_diversity(
        results: List[RetrievedDocument],
        max_per_section: int = 2
    ) -> List[RetrievedDocument]:
        """Ensure diverse results across sections"""
        filtered = []
        section_counts = {}
        
        for doc in results:
            section = doc.metadata.get("section_title", "Other")
            count = section_counts.get(section, 0)
            
            if count < max_per_section:
                filtered.append(doc)
                section_counts[section] = count + 1
        
        return filtered


def retrieve_legal_documents(
    query: str,
    k: int = 5,
    method: str = "hybrid",
    expand_query: bool = True,
    rerank: bool = True,
    filter_config: Optional[Dict[str, Any]] = None
) -> List[RetrievedDocument]:
    """
    Convenience function for legal document retrieval
    
    Args:
        query: User query
        k: Number of results
        method: "semantic", "keyword", or "hybrid"
        expand_query: Whether to expand query with legal terms
        rerank: Whether to rerank results
        filter_config: Optional filtering configuration
        
    Returns:
        List of retrieved documents
    """
    # Create retriever
    retriever = HybridRetriever()
    
    # Convert string method to enum
    search_method = SearchMethod[method.upper()]
    
    # Retrieve
    results = retriever.retrieve(
        query=query,
        k=k,
        method=search_method,
        query_expansion=expand_query,
        rerank=rerank
    )
    
    # Apply filters if provided
    if filter_config:
        filtering = AdvancedFiltering()
        
        if "section_title" in filter_config:
            results = filtering.filter_by_section(results, filter_config["section_title"])
        
        if "chunk_type" in filter_config:
            results = filtering.filter_by_chunk_type(results, filter_config["chunk_type"])
        
        if "document_id" in filter_config:
            results = filtering.filter_by_document(results, filter_config["document_id"])
        
        if "relevance_threshold" in filter_config:
            results = filtering.filter_by_relevance_threshold(
                results, filter_config["relevance_threshold"]
            )
    
    return results
