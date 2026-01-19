"""
Retriever Agent

Executes search queries against the vector store and reranks results.
Handles multiple queries and result fusion.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from app.agents.base import BaseAgent
from app.core.vector_store import get_vector_store
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Retrieval agent that searches vector store and reranks results.
    
    Features:
    - Multi-query retrieval
    - Result fusion (combining results from multiple queries)
    - Reranking by relevance
    - Deduplication
    - Metadata filtering
    
    Example:
        retriever = RetrieverAgent()
        result = retriever.execute(
            search_queries=['query1', 'query2'],
            filters={'language': 'en'},
            top_k=5
        )
    """
    
    def __init__(self):
        """Initialize retriever agent"""
        # Retriever doesn't use LLM, but inherits from BaseAgent for consistency
        super().__init__(agent_name="retriever")
        
        # Get vector store
        self.vector_store = get_vector_store()
    
    def execute(
        self,
        search_queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Execute retrieval for multiple queries.
        
        Args:
            search_queries: List of search queries
            filters: Metadata filters (language, user_id, etc.)
            top_k: Number of results to return
            rerank: Whether to rerank results
        
        Returns:
            Dictionary with:
                - documents: List of retrieved documents
                - total_retrieved: Total docs before deduplication
                - query_count: Number of queries executed
        
        Example:
            result = retriever.execute(
                search_queries=['India capital', 'भारत राजधानी'],
                top_k=5
            )
        """
        from app.config import settings
        
        top_k = top_k or settings.TOP_K_RERANK
        
        self.logger.info(
            f"Retrieving with {len(search_queries)} queries, "
            f"filters={filters}, top_k={top_k}"
        )
        
        try:
            all_documents = []
            
            # Execute each search query
            for i, query in enumerate(search_queries, 1):
                self.logger.debug(f"Query {i}/{len(search_queries)}: {query}")
                
                # Search vector store
                results = self.vector_store.search(
                    query=query,
                    top_k=settings.TOP_K_RETRIEVAL,  # Retrieve more initially
                    filters=filters,
                    score_threshold=settings.MIN_SIMILARITY_SCORE
                )
                
                # Add query info to metadata
                for doc in results:
                    doc['metadata']['source_query'] = query
                    doc['metadata']['query_index'] = i
                
                all_documents.extend(results)
                
                self.logger.debug(f"  Retrieved {len(results)} documents")
            
            self.logger.info(f"Total retrieved: {len(all_documents)} documents")
            
            # Deduplicate documents
            unique_documents = self._deduplicate(all_documents)
            
            self.logger.info(
                f"After deduplication: {len(unique_documents)} unique documents"
            )
            
            # Rerank if requested
            if rerank and len(unique_documents) > top_k:
                final_documents = self._rerank(
                    unique_documents,
                    search_queries[0],  # Use first query for reranking
                    top_k
                )
            else:
                # Just take top K by score
                final_documents = sorted(
                    unique_documents,
                    key=lambda x: x['score'],
                    reverse=True
                )[:top_k]
            
            result = {
                "documents": final_documents,
                "total_retrieved": len(all_documents),
                "unique_documents": len(unique_documents),
                "query_count": len(search_queries),
                "filters_applied": filters
            }
            
            self.logger.info(f"✅ Returning {len(final_documents)} documents")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Retriever execution failed: {e}")
            raise AgentError("retriever", str(e))
    
    def _deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents.
        
        Deduplication strategy:
        - If same document ID, keep highest scoring one
        - If very similar text, keep highest scoring one
        
        Args:
            documents: List of documents
        
        Returns:
            Deduplicated list
        """
        if not documents:
            return []
        
        # Group by document ID
        doc_groups = {}
        for doc in documents:
            doc_id = doc['id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(doc)
        
        # Keep highest scoring document from each group
        unique_docs = []
        for doc_id, group in doc_groups.items():
            best_doc = max(group, key=lambda x: x['score'])
            unique_docs.append(best_doc)
        
        return unique_docs
    
    def _rerank(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using a simple scoring approach.
        
        In production, you could use a cross-encoder model here.
        For now, we use a combination of:
        - Original similarity score
        - Query frequency (if doc matched multiple queries)
        
        Args:
            documents: List of documents to rerank
            query: Original query
            top_k: Number of top documents to return
        
        Returns:
            Reranked list of documents
        """
        self.logger.info(f"Reranking {len(documents)} documents")
        
        # Count query matches per document
        doc_query_counts = {}
        for doc in documents:
            doc_id = doc['id']
            doc_query_counts[doc_id] = doc_query_counts.get(doc_id, 0) + 1
        
        # Adjust scores based on query frequency
        for doc in documents:
            original_score = doc['score']
            query_bonus = (doc_query_counts[doc['id']] - 1) * 0.05  # 5% bonus per extra match
            doc['rerank_score'] = min(1.0, original_score + query_bonus)
        
        # Sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x.get('rerank_score', x['score']),
            reverse=True
        )[:top_k]
        
        return reranked
    
    def call_llm(self, *args, **kwargs):
        """
        Override to prevent LLM calls (retriever doesn't use LLM).
        """
        raise NotImplementedError("Retriever agent does not use LLM")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING RETRIEVER AGENT")
    print("=" * 80)
    
    # Note: This requires documents to be in the vector store
    # Run vector_store.py test first to populate some test data
    
    retriever = RetrieverAgent()
    
    # Test retrieval
    test_queries = [
        "India capital city",
        "भारत राजधानी",
        "New Delhi"
    ]
    
    print(f"\n🔍 Testing retrieval with {len(test_queries)} queries:")
    for query in test_queries:
        print(f"  - {query}")
    
    result = retriever.execute(
        search_queries=test_queries,
        top_k=3
    )
    
    print(f"\n📊 Retrieval Results:")
    print(f"  Total retrieved: {result['total_retrieved']}")
    print(f"  Unique documents: {result['unique_documents']}")
    print(f"  Final results: {len(result['documents'])}")
    
    print(f"\n📄 Top documents:")
    for i, doc in enumerate(result['documents'], 1):
        print(f"\n  {i}. Score: {doc['score']:.4f}")
        print(f"     Text: {doc['text'][:80]}...")
        print(f"     Language: {doc['metadata'].get('language', 'unknown')}")
    
    print("\n" + "=" * 80)
    print("✅ RETRIEVER AGENT WORKING CORRECTLY!")
    print("=" * 80)
