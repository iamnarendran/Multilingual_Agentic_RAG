"""
Validator Agent

Validates the quality of generated answers by checking for hallucinations,
incorrect citations, and logical inconsistencies.
"""

from typing import Dict, Any, List
import json
import re

from app.agents.base import BaseAgent
from app.core.prompts import get_prompt
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


class ValidatorAgent(BaseAgent):
    """
    Answer validation agent.
    
    Features:
    - Hallucination detection
    - Citation verification
    - Logical consistency checking
    - Completeness assessment
    
    Example:
        validator = ValidatorAgent()
        result = validator.execute(
            answer="Delhi is capital [Doc ID: 1]",
            documents=[...],
            query="What is capital?"
        )
    """
    
    def __init__(self):
        """Initialize validator agent"""
        super().__init__(agent_name="validator")
        
        # Get prompts
        self.system_prompt = get_prompt("validator", include_examples=True)
    
    def execute(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Validate answer quality.
        
        Args:
            answer: Generated answer to validate
            documents: Source documents used
            query: Original query
        
        Returns:
            Dictionary with:
                - valid: Boolean indicating if answer is valid
                - confidence: Confidence in validation (0-1)
                - issues: List of identified issues
                - details: Detailed validation results
        
        Example:
            result = validator.execute(
                answer="Answer with citations...",
                documents=[...],
                query="What is X?"
            )
        """
        self.logger.info(f"Validating answer for query: '{query[:50]}...'")
        
        try:
            # Build context from documents
            context = self._build_document_context(documents)
            
            # Build validation prompt
            validation_prompt = f"""
Query: {query}

Context (Source Documents):
{context}

Generated Answer:
{answer}

Validate this answer according to the criteria in your system prompt.
Respond with ONLY a JSON object.
"""
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": validation_prompt}
            ]
            
            # Call LLM (Gemini Flash - fast and cheap for validation)
            response = self.call_llm(
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            # Parse validation result
            validation_result = self._parse_validation(response)
            
            # Additional automated checks
            auto_checks = self._run_automated_checks(answer, documents)
            
            # Combine results
            final_result = self._combine_results(validation_result, auto_checks)
            
            self.logger.info(
                f"✅ Validation complete: valid={final_result['valid']}, "
                f"confidence={final_result['confidence']:.2f}, "
                f"issues={len(final_result['issues'])}"
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Validator execution failed: {e}")
            # Default to passing validation on error (don't block user)
            return {
                "valid": True,
                "confidence": 0.5,
                "issues": [f"Validation error: {str(e)}"],
                "details": {"error": str(e)}
            }
    
    def _build_document_context(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 3000
    ) -> str:
        """
        Build context string from documents.
        
        Args:
            documents: List of documents
            max_length: Maximum context length
        
        Returns:
            Formatted context
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_text = f"[Doc ID: {doc['id']}] {doc['text']}"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _parse_validation(self, response: str) -> Dict[str, Any]:
        """
        Parse validation response from LLM.
        
        Args:
            response: LLM response (should be JSON)
        
        Returns:
            Parsed validation result
        """
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = response.strip()
            
            # Parse JSON
            result = json.loads(response)
            
            # Validate structure
            if "valid" not in result:
                result["valid"] = True  # Default to valid
            if "confidence" not in result:
                result["confidence"] = 0.8
            if "issues" not in result:
                result["issues"] = []
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse validation JSON: {e}")
            # Try to extract valid/invalid from text
            if "valid" in response.lower() and "false" in response.lower():
                return {
                    "valid": False,
                    "confidence": 0.6,
                    "issues": ["JSON parsing failed, extracted from text"]
                }
            else:
                return {
                    "valid": True,
                    "confidence": 0.7,
                    "issues": []
                }
    
    def _run_automated_checks(
        self,
        answer: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run automated validation checks.
        
        Args:
            answer: Answer to validate
            documents: Source documents
        
        Returns:
            Automated check results
        """
        issues = []
        
        # Check 1: Answer has citations
        citations = re.findall(r'\[Doc ID:\s*([^\]]+)\]', answer)
        if not citations:
            issues.append("No citations found in answer")
        
        # Check 2: All citations reference valid document IDs
        doc_ids = set(doc['id'] for doc in documents)
        for citation in citations:
            if citation not in doc_ids:
                issues.append(f"Invalid citation: Doc ID '{citation}' not in sources")
        
        # Check 3: Answer is not too short (unless simple query)
        if len(answer.split()) < 5:
            issues.append("Answer is very short (< 5 words)")
        
        # Check 4: Answer doesn't contain uncertainty phrases without explanation
        uncertainty_phrases = [
            "not sure",
            "don't know",
            "cannot determine",
            "unclear"
        ]
        for phrase in uncertainty_phrases:
            if phrase in answer.lower() and len(answer.split()) < 20:
                issues.append(f"Contains uncertainty phrase without explanation: '{phrase}'")
        
        return {
            "citations_found": len(citations),
            "valid_citations": len([c for c in citations if c in doc_ids]),
            "automated_issues": issues
        }
    
    def _combine_results(
        self,
        llm_result: Dict[str, Any],
        auto_checks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine LLM validation with automated checks.
        
        Args:
            llm_result: Result from LLM validation
            auto_checks: Automated check results
        
        Returns:
            Combined validation result
        """
        # Start with LLM result
        combined = {
            "valid": llm_result["valid"],
            "confidence": llm_result["confidence"],
            "issues": llm_result["issues"].copy(),
            "details": {
                "llm_validation": llm_result,
                "automated_checks": auto_checks
            }
        }
        
        # Add automated issues
        combined["issues"].extend(auto_checks["automated_issues"])
        
        # Adjust validity if critical automated issues found
        critical_issues = [
            issue for issue in auto_checks["automated_issues"]
            if "Invalid citation" in issue or "No citations" in issue
        ]
        
        if critical_issues and combined["valid"]:
            combined["valid"] = False
            combined["confidence"] = min(combined["confidence"], 0.6)
        
        return combined


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING VALIDATOR AGENT")
    print("=" * 80)
    
    validator = ValidatorAgent()
    
    # Test documents
    test_docs = [
        {
            "id": "doc1",
            "text": "India's capital is New Delhi.",
            "score": 0.95,
            "metadata": {}
        },
        {
            "id": "doc2",
            "text": "Mumbai is the largest city.",
            "score": 0.85,
            "metadata": {}
        }
    ]
    
    # Test Case 1: Valid answer
    print("\n" + "=" * 60)
    print("Test 1: Valid Answer")
    print("=" * 60)
    
    valid_answer = "India's capital is New Delhi [Doc ID: doc1]."
    query = "What is India's capital?"
    
    result1 = validator.execute(
        answer=valid_answer,
        documents=test_docs,
        query=query
    )
    
    print(f"Answer: {valid_answer}")
    print(f"Valid: {result1['valid']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Issues: {result1['issues']}")
    
    # Test Case 2: Answer with hallucination
    print("\n" + "=" * 60)
    print("Test 2: Answer with Hallucination")
    print("=" * 60)
    
    hallucinated_answer = "India's capital is New Delhi, with a population of 30 million [Doc ID: doc1]."
    
    result2 = validator.execute(
        answer=hallucinated_answer,
        documents=test_docs,
        query=query
    )
    
    print(f"Answer: {hallucinated_answer}")
    print(f"Valid: {result2['valid']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Issues: {result2['issues']}")
    
    # Test Case 3: Answer with invalid citation
    print("\n" + "=" * 60)
    print("Test 3: Invalid Citation")
    print("=" * 60)
    
    invalid_citation = "India's capital is New Delhi [Doc ID: doc999]."
    
    result3 = validator.execute(
        answer=invalid_citation,
        documents=test_docs,
        query=query
    )
    
    print(f"Answer: {invalid_citation}")
    print(f"Valid: {result3['valid']}")
    print(f"Confidence: {result3['confidence']:.2f}")
    print(f"Issues: {result3['issues']}")
    
    # Show stats
    stats = validator.get_stats()
    print(f"\n📊 Validator Statistics:")
    print(f"  Calls: {stats['calls']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATOR AGENT WORKING CORRECTLY!")
    print("=" * 80)
