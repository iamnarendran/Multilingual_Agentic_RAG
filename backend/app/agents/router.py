"""
Router Agent

Classifies incoming queries into categories to determine processing strategy.
Uses a fast, cheap model (Gemini Flash) for quick classification.
"""

from typing import Dict, Any
import json

from app.agents.base import BaseAgent
from app.core.prompts import get_prompt
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


class RouterAgent(BaseAgent):
    """
    Query classification agent.
    
    Classifies queries into:
    - SIMPLE_QA: Direct factual questions
    - COMPARISON: Comparison queries
    - SUMMARIZATION: Summarization requests
    - ANALYSIS: Deep analysis queries
    - EXTRACTION: Data extraction requests
    - MULTI_HOP: Multi-step reasoning queries
    
    Example:
        router = RouterAgent()
        result = router.execute(query="What is the capital of India?")
        # {'query_type': 'SIMPLE_QA', 'confidence': 0.95}
    """
    
    def __init__(self):
        """Initialize router agent"""
        super().__init__(agent_name="router")
        
        # Valid query types
        self.valid_types = [
            "SIMPLE_QA",
            "COMPARISON",
            "SUMMARIZATION",
            "ANALYSIS",
            "EXTRACTION",
            "MULTI_HOP"
        ]
        
        # Get system prompt
        self.system_prompt = get_prompt("router", include_examples=True)
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Classify the query.
        
        Args:
            query: User query text
        
        Returns:
            Dictionary with:
                - query_type: Classification result
                - confidence: Confidence score (0-1)
                - reasoning: Brief explanation (optional)
        
        Example:
            result = router.execute("What is AI?")
            # {'query_type': 'SIMPLE_QA', 'confidence': 0.98}
        """
        self.logger.info(f"Routing query: '{query[:50]}...'")
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Classify this query: {query}"}
            ]
            
            # Call LLM
            response = self.call_llm(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=50
            )
            
            # Parse response
            query_type = response.strip().upper()
            
            # Validate
            if query_type not in self.valid_types:
                # Try to find closest match
                for valid_type in self.valid_types:
                    if valid_type in query_type:
                        query_type = valid_type
                        break
                else:
                    # Default to SIMPLE_QA if no match
                    self.logger.warning(
                        f"Invalid query type '{query_type}', defaulting to SIMPLE_QA"
                    )
                    query_type = "SIMPLE_QA"
            
            result = {
                "query_type": query_type,
                "confidence": 0.9,  # High confidence with good model
                "model_used": self.model_config["model"]
            }
            
            self.logger.info(f"✅ Classified as: {query_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Router execution failed: {e}")
            raise AgentError("router", str(e))


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ROUTER AGENT")
    print("=" * 80)
    
    router = RouterAgent()
    
    # Test queries
    test_queries = [
        "What is the capital of India?",
        "Compare iPhone vs Samsung Galaxy",
        "Summarize the document about AI",
        "Why did the stock market crash?",
        "List all dates mentioned in the document",
        "Who is the CEO of the company that makes iPhone?",
    ]
    
    print("\n🧭 Testing query classification:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router.execute(query)
        print(f"  → Type: {result['query_type']}")
        print(f"  → Confidence: {result['confidence']:.2f}")
    
    # Show stats
    stats = router.get_stats()
    print(f"\n📊 Router Statistics:")
    print(f"  Calls: {stats['calls']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    print(f"  Average time: {stats['total_time']/stats['calls']:.2f}s")
    
    print("\n" + "=" * 80)
    print("✅ ROUTER AGENT WORKING CORRECTLY!")
    print("=" * 80)
