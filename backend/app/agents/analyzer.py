"""
Analyzer Agent

Analyzes retrieved documents and extracts relevant information to answer queries.
Uses Claude Sonnet for deep reasoning and accurate extraction.
"""

from typing import Dict, Any, List
import json

from app.agents.base import BaseAgent
from app.core.prompts import (
    get_prompt,
    build_context_with_docs,
    build_rag_prompt
)
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError
from app.utils.helpers import extract_citations

logger = get_logger(__name__)


class AnalyzerAgent(BaseAgent):
    """
    Document analysis agent.
    
    Features:
    - Deep reading of retrieved documents
    - Citation tracking
    - Multi-lingual understanding
    - Accurate information extraction
    
    Example:
        analyzer = AnalyzerAgent()
        result = analyzer.execute(
            query="What is India's capital?",
            documents=[...],
            language="en"
        )
    """
    
    def __init__(self):
        """Initialize analyzer agent"""
        super().__init__(agent_name="analyzer")
        
        # Get prompts
        self.system_prompt = get_prompt("analyzer", include_examples=True)
    
    def execute(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        language: str = "en",
        query_type: str = "SIMPLE_QA"
    ) -> Dict[str, Any]:
        """
        Analyze documents to answer query.
        
        Args:
            query: User query
            documents: Retrieved documents from vector store
            language: Query language
            query_type: Type of query (for context)
        
        Returns:
            Dictionary with:
                - analysis: Extracted information with citations
                - sources_used: List of document IDs used
                - confidence: Confidence in the analysis
                - citations_count: Number of citations made
        
        Example:
            result = analyzer.execute(
                query="What is the capital?",
                documents=[{'id': '1', 'text': 'Capital is Delhi', ...}]
            )
        """
        self.logger.info(f"Analyzing {len(documents)} documents for query")
        
        if not documents:
            return {
                "analysis": "No documents provided for analysis.",
                "sources_used": [],
                "confidence": 0.0,
                "citations_count": 0
            }
        
        try:
            # Build context from documents
            context = build_context_with_docs(documents, max_tokens=6000)
            
            # Build complete prompt
            full_prompt = build_rag_prompt(
                query=query,
                context=context,
                system_prompt=self.system_prompt,
                few_shot_examples=None  # Already in system prompt
            )
            
            # Add language instruction if not English
            if language != "en":
                full_prompt += f"\n\nIMPORTANT: Respond in {language}."
            
            # Build messages
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            # Call LLM (Claude Sonnet for best reasoning)
            analysis = self.call_llm(
                messages=messages,
                temperature=0.3,  # Lower for accuracy
                max_tokens=2000
            )
            
            # Extract citations
            citations = extract_citations(analysis)
            sources_used = list(set([c['doc_id'] for c in citations]))
            
            # Estimate confidence based on citations and document scores
            confidence = self._estimate_confidence(
                analysis,
                citations,
                documents
            )
            
            result = {
                "analysis": analysis,
                "sources_used": sources_used,
                "confidence": confidence,
                "citations_count": len(citations),
                "documents_analyzed": len(documents),
                "model_used": self.model_config["model"]
            }
            
            self.logger.info(
                f"✅ Analysis complete: {len(citations)} citations, "
                f"confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analyzer execution failed: {e}")
            raise AgentError("analyzer", str(e))
    
    def _estimate_confidence(
        self,
        analysis: str,
        citations: List[Dict[str, Any]],
        documents: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate confidence in the analysis.
        
        Factors:
        - Number of citations (more = better)
        - Document relevance scores
        - Presence of uncertainty phrases
        
        Args:
            analysis: Analysis text
            citations: Extracted citations
            documents: Source documents
        
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Citations (up to +0.3)
        if len(citations) > 0:
            citation_bonus = min(0.3, len(citations) * 0.1)
            confidence += citation_bonus
        
        # Factor 2: Document relevance (up to +0.2)
        if documents:
            avg_score = sum(doc.get('score', 0) for doc in documents) / len(documents)
            confidence += avg_score * 0.2
        
        # Factor 3: Uncertainty phrases (up to -0.3)
        uncertainty_phrases = [
            "not found in provided documents",
            "information not available",
            "unclear",
            "uncertain",
            "may be",
            "possibly"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in analysis.lower():
                confidence -= 0.1
        
        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ANALYZER AGENT")
    print("=" * 80)
    
    analyzer = AnalyzerAgent()
    
    # Test documents
    test_docs = [
        {
            "id": "doc1",
            "text": "India's capital is New Delhi, located in the northern part of the country.",
            "score": 0.95,
            "metadata": {"document_name": "india_facts.pdf", "language": "en"}
        },
        {
            "id": "doc2",
            "text": "भारत की राजधानी नई दिल्ली है। यह देश के उत्तरी भाग में स्थित है।",
            "score": 0.92,
            "metadata": {"document_name": "bharat_tathya.pdf", "language": "hi"}
        }
    ]
    
    # Test query
    query = "What is the capital of India?"
    
    print(f"\n📝 Query: {query}")
    print(f"📚 Analyzing {len(test_docs)} documents...")
    
    result = analyzer.execute(
        query=query,
        documents=test_docs,
        language="en",
        query_type="SIMPLE_QA"
    )
    
    print(f"\n✅ Analysis Results:")
    print(f"{'=' * 60}")
    print(result['analysis'])
    print(f"{'=' * 60}")
    
    print(f"\n📊 Metadata:")
    print(f"  Sources used: {result['sources_used']}")
    print(f"  Citations count: {result['citations_count']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Documents analyzed: {result['documents_analyzed']}")
    
    # Show stats
    stats = analyzer.get_stats()
    print(f"\n📊 Analyzer Statistics:")
    print(f"  Calls: {stats['calls']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYZER AGENT WORKING CORRECTLY!")
    print("=" * 80)
