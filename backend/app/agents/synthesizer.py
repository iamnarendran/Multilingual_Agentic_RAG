"""
Synthesizer Agent

Combines information from analysis into a coherent final answer.
Resolves contradictions and structures the response.
"""

from typing import Dict, Any, List, Optional
import json

from app.agents.base import BaseAgent
from app.core.prompts import get_prompt
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError
from app.utils.helpers import extract_citations

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Answer synthesis agent.
    
    Features:
    - Combines multiple analyses
    - Resolves contradictions
    - Structures coherent responses
    - Maintains citations
    - Language-aware formatting
    
    Example:
        synthesizer = SynthesizerAgent()
        result = synthesizer.execute(
            query="What is the capital?",
            analysis="Analysis with citations...",
            language="en"
        )
    """
    
    def __init__(self):
        """Initialize synthesizer agent"""
        super().__init__(agent_name="synthesizer")
        
        # Get prompts
        self.system_prompt = get_prompt("synthesizer", include_examples=True)
    
    def execute(
        self,
        query: str,
        analysis: str,
        language: str = "en",
        query_type: str = "SIMPLE_QA",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize final answer from analysis.
        
        Args:
            query: Original user query
            analysis: Analysis from analyzer agent
            language: Response language
            query_type: Type of query
            additional_context: Optional additional context
        
        Returns:
            Dictionary with:
                - answer: Final synthesized answer
                - citations: List of citations
                - structure: Answer structure (direct/detailed/comparison)
                - confidence: Overall confidence
        
        Example:
            result = synthesizer.execute(
                query="What is AI?",
                analysis="AI is... [Doc ID: 1]"
            )
        """
        self.logger.info(f"Synthesizing answer for query type: {query_type}")
        
        try:
            # Build synthesis prompt
            user_prompt = self._build_synthesis_prompt(
                query=query,
                analysis=analysis,
                query_type=query_type,
                language=language,
                additional_context=additional_context
            )
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM (Claude Sonnet for best synthesis)
            final_answer = self.call_llm(
                messages=messages,
                temperature=0.2,  # Low for consistency
                max_tokens=1500
            )
            
            # Extract citations from final answer
            citations = extract_citations(final_answer)
            
            # Determine answer structure
            structure = self._determine_structure(final_answer, query_type)
            
            result = {
                "answer": final_answer,
                "citations": citations,
                "structure": structure,
                "query_type": query_type,
                "language": language,
                "model_used": self.model_config["model"]
            }
            
            self.logger.info(
                f"✅ Synthesis complete: {len(citations)} citations, "
                f"structure={structure}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Synthesizer execution failed: {e}")
            raise AgentError("synthesizer", str(e))
    
    def _build_synthesis_prompt(
        self,
        query: str,
        analysis: str,
        query_type: str,
        language: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for synthesis.
        
        Args:
            query: User query
            analysis: Analysis text
            query_type: Query type
            language: Target language
            additional_context: Additional context
        
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            f"Original Query: {query}",
            f"Query Type: {query_type}",
            "",
            "Analysis to Synthesize:",
            "=" * 60,
            analysis,
            "=" * 60,
            "",
            "Your Task:",
            "1. Combine the analysis into a coherent answer",
            "2. Maintain ALL citations from the analysis",
            "3. Resolve any contradictions (note disagreements)",
            "4. Structure appropriately for the query type",
        ]
        
        # Add query-type specific instructions
        if query_type == "COMPARISON":
            prompt_parts.append("5. Use comparison structure (similarities, differences)")
        elif query_type == "SUMMARIZATION":
            prompt_parts.append("5. Provide structured summary with key points")
        elif query_type == "ANALYSIS":
            prompt_parts.append("5. Provide reasoning and implications")
        else:
            prompt_parts.append("5. Start with direct answer, then details")
        
        # Add language instruction
        if language != "en":
            prompt_parts.append(f"\nIMPORTANT: Respond in {language}.")
        
        # Add additional context if provided
        if additional_context:
            prompt_parts.append(f"\nAdditional Context: {additional_context}")
        
        return "\n".join(prompt_parts)
    
    def _determine_structure(self, answer: str, query_type: str) -> str:
        """
        Determine the structure of the answer.
        
        Args:
            answer: Final answer text
            query_type: Type of query
        
        Returns:
            Structure type (direct/detailed/comparison/summary)
        """
        # Check length
        word_count = len(answer.split())
        
        if word_count < 30:
            return "direct"
        elif query_type == "COMPARISON":
            return "comparison"
        elif query_type == "SUMMARIZATION":
            return "summary"
        elif word_count > 100:
            return "detailed"
        else:
            return "standard"


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING SYNTHESIZER AGENT")
    print("=" * 80)
    
    synthesizer = SynthesizerAgent()
    
    # Test analysis (from analyzer)
    test_analysis = """
    Based on the provided documents:
    
    India's capital is New Delhi [Doc ID: doc1]. It is located in the northern 
    part of the country [Doc ID: doc1]. The Hindi translation confirms this: 
    भारत की राजधानी नई दिल्ली है [Doc ID: doc2].
    
    Mumbai is mentioned as the financial capital and largest city [Doc ID: doc1], 
    but it is not the administrative capital.
    """
    
    # Test query
    query = "What is the capital of India?"
    
    print(f"\n📝 Query: {query}")
    print(f"🔄 Synthesizing from analysis...")
    
    result = synthesizer.execute(
        query=query,
        analysis=test_analysis,
        language="en",
        query_type="SIMPLE_QA"
    )
    
    print(f"\n✅ Final Answer:")
    print(f"{'=' * 60}")
    print(result['answer'])
    print(f"{'=' * 60}")
    
    print(f"\n📊 Metadata:")
    print(f"  Citations: {len(result['citations'])}")
    print(f"  Structure: {result['structure']}")
    print(f"  Query type: {result['query_type']}")
    print(f"  Language: {result['language']}")
    
    # Test with different query type
    print(f"\n{'=' * 80}")
    print("Testing COMPARISON query type:")
    
    comparison_analysis = """
    iPhone uses iOS [Doc ID: 1] while Samsung uses Android [Doc ID: 2].
    iPhone has better app ecosystem [Doc ID: 1]. Samsung offers more 
    customization [Doc ID: 2]. Both have excellent cameras [Doc ID: 3].
    """
    
    result2 = synthesizer.execute(
        query="Compare iPhone vs Samsung",
        analysis=comparison_analysis,
        query_type="COMPARISON"
    )
    
    print(f"\n✅ Comparison Answer:")
    print(f"{'=' * 60}")
    print(result2['answer'])
    print(f"{'=' * 60}")
    
    # Show stats
    stats = synthesizer.get_stats()
    print(f"\n📊 Synthesizer Statistics:")
    print(f"  Calls: {stats['calls']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ SYNTHESIZER AGENT WORKING CORRECTLY!")
    print("=" * 80)
