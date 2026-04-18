"""
Vector Store Management for Legal Documents
Handles embeddings, storage, and retrieval
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from ingestion.chunking import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages embeddings and vector storage using Chroma
    """
    
    def __init__(self, collection_name: str = None, path: str = None):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the Chroma collection
            path: Path to store Chroma database
        """
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.path = path or settings.CHROMA_DB_PATH
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.path
        )
        
        logger.info(f"Initialized vector store at {self.path}")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to add at once
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        doc_ids = []
        
        # Add in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                ids = self.vector_store.add_documents(documents=batch)
                doc_ids.extend(ids)
                logger.info(f"Added batch {i//batch_size + 1}, {len(ids)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                raise
        
        # Persist to disk
        self.vector_store.persist()
        logger.info(f"Added {len(doc_ids)} documents total")
        
        return doc_ids
    
    def add_chunks(
        self,
        chunks: List['Chunk'],
        document_id: Optional[str] = None
    ) -> List[str]:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of Chunk objects
            document_id: Optional ID for the parent document
            
        Returns:
            List of chunk IDs
        """
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy() if chunk.metadata else {}
            
            # Add chunk information to metadata
            metadata.update({
                "chunk_index": i,
                "section_title": chunk.section_title or "Unnamed",
                "section_number": chunk.section_number or "",
                "chunk_type": chunk.chunk_type,
                "timestamp": datetime.now().isoformat(),
            })
            
            if document_id:
                metadata["document_id"] = document_id
            
            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            documents.append(doc)
        
        return self.add_documents(documents)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search vector store for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with content and metadata
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1 - score,  # Convert distance to similarity
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from vector store
        
        Args:
            document_ids: IDs of documents to delete
            
        Returns:
            Success status
        """
        try:
            for doc_id in document_ids:
                self.vector_store.delete([doc_id])
            
            self.vector_store.persist()
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection = self.vector_store._collection
            return {
                "name": collection.name,
                "count": collection.count(),
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            collection = self.vector_store._collection
            collection.delete(where={})  # Delete all documents
            self.vector_store.persist()
            logger.info("Cleared collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


def create_vector_store(
    collection_name: str = None,
    path: str = None
) -> VectorStore:
    """
    Factory function to create vector store
    """
    return VectorStore(collection_name=collection_name, path=path)
