"""
RAG Orchestrator using LangGraph

Coordinates all agents in a stateful workflow to process queries end-to-end.
Uses LangGraph for complex multi-agent orchestration.
"""

from typing import Dict, Any, TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
import time

from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.analyzer import AnalyzerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.validator import ValidatorAgent
from app.core.language_detector import detect_with_confidence
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class RAGState(TypedDict):
    """
    State object passed between agents in the workflow.
    
    Each agent reads from and writes to this state.
    """
    # Input
    query: str
    user_id: str
    filters: Dict[str, Any]
    
    # Language detection
    language: str
    language_confidence: float
    
    # Router output
    query_type: str
    
    # Planner output
    search_queries: List[str]
    
    # Retriever output
    documents: List[Dict[str, Any]]
    total_retrieved: int
    
    # Analyzer output
    analysis: str
    sources_used: List[str]
    analysis_confidence: float
    
    # Synthesizer output
    answer: str
    citations: List[Dict[str, Any]]
    
    # Validator output
    validation_valid: bool
    validation_confidence: float
    validation_issues: List[str]
    
    # Metadata
    processing_time: float
    total_cost: float
    agent_stats: Dict[str, Any]
    errors: Annotated[List[str], operator.add]


# =============================================================================
# RAG ORCHESTRATOR
# =============================================================================

class RAGOrchestrator:
    """
    Main orchestrator for the RAG system.
    
    Coordinates all agents using LangGraph to process queries from start to finish.
    
    Workflow:
    1. Detect language
    2. Route query (classify type)
    3. Plan queries (generate search queries)
    4. Retrieve documents
    5. Analyze documents
    6. Synthesize answer
    7. Validate answer

Example:
    orchestrator = RAGOrchestrator()
    result = orchestrator.process_query(
        query="What is the capital of India?",
        user_id="user123"
    )
"""

def __init__(self):
    """Initialize orchestrator and all agents"""
    logger.info("Initializing RAG Orchestrator...")
    
    # Initialize all agents
    self.router = RouterAgent()
    self.planner = PlannerAgent()
    self.retriever = RetrieverAgent()
    self.analyzer = AnalyzerAgent()
    self.synthesizer = SynthesizerAgent()
    self.validator = ValidatorAgent()
    
    # Build LangGraph workflow
    self.workflow = self._build_workflow()
    
    logger.info("✅ RAG Orchestrator initialized with all agents")

def _build_workflow(self) -> StateGraph:
    """
    Build the LangGraph workflow.
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes (each node is an agent execution)
    workflow.add_node("detect_language", self._detect_language_node)
    workflow.add_node("route", self._route_node)
    workflow.add_node("plan", self._plan_node)
    workflow.add_node("retrieve", self._retrieve_node)
    workflow.add_node("analyze", self._analyze_node)
    workflow.add_node("synthesize", self._synthesize_node)
    workflow.add_node("validate", self._validate_node)
    
    # Define edges (workflow flow)
    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "route")
    workflow.add_edge("route", "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "synthesize")
    workflow.add_edge("synthesize", "validate")
    workflow.add_edge("validate", END)
    
    # Compile workflow
    return workflow.compile()

# =========================================================================
# NODE FUNCTIONS (Each represents an agent execution)
# =========================================================================

def _detect_language_node(self, state: RAGState) -> RAGState:
    """Detect query language"""
    logger.info("🌍 Node: Detecting language...")
    
    lang, confidence = detect_with_confidence(state["query"])
    
    state["language"] = lang
    state["language_confidence"] = confidence
    
    logger.info(f"✅ Language: {lang} (confidence: {confidence:.2f})")
    
    return state

def _route_node(self, state: RAGState) -> RAGState:
    """Route/classify query"""
    logger.info("🧭 Node: Routing query...")
    
    result = self.router.execute(query=state["query"])
    
    state["query_type"] = result["query_type"]
    
    logger.info(f"✅ Query type: {result['query_type']}")
    
    return state

def _plan_node(self, state: RAGState) -> RAGState:
    """Plan search queries"""
    logger.info("🗓️  Node: Planning queries...")
    
    result = self.planner.execute(
        query=state["query"],
        query_type=state["query_type"],
        num_queries=3
    )
    
    state["search_queries"] = result["search_queries"]
    
    logger.info(f"✅ Generated {len(result['search_queries'])} search queries")
    
    return state

def _retrieve_node(self, state: RAGState) -> RAGState:
    """Retrieve documents"""
    logger.info("🔍 Node: Retrieving documents...")
    
    result = self.retriever.execute(
        search_queries=state["search_queries"],
        filters=state.get("filters"),
        top_k=5
    )
    
    state["documents"] = result["documents"]
    state["total_retrieved"] = result["total_retrieved"]
    
    logger.info(f"✅ Retrieved {len(result['documents'])} documents")
    
    return state

def _analyze_node(self, state: RAGState) -> RAGState:
    """Analyze documents"""
    logger.info("📊 Node: Analyzing documents...")
    
    result = self.analyzer.execute(
        query=state["query"],
        documents=state["documents"],
        language=state["language"],
        query_type=state["query_type"]
    )
    
    state["analysis"] = result["analysis"]
    state["sources_used"] = result["sources_used"]
    state["analysis_confidence"] = result["confidence"]
    
    logger.info(f"✅ Analysis complete with {result['citations_count']} citations")
    
    return state

def _synthesize_node(self, state: RAGState) -> RAGState:
    """Synthesize final answer"""
    logger.info("🔄 Node: Synthesizing answer...")
    
    result = self.synthesizer.execute(
        query=state["query"],
        analysis=state["analysis"],
        language=state["language"],
        query_type=state["query_type"]
    )
    
    state["answer"] = result["answer"]
    state["citations"] = result["citations"]
    
    logger.info(f"✅ Synthesis complete")
    
    return state

def _validate_node(self, state: RAGState) -> RAGState:
    """Validate answer"""
    logger.info("✓ Node: Validating answer...")
    
    result = self.validator.execute(
        answer=state["answer"],
        documents=state["documents"],
        query=state["query"]
    )
    
    state["validation_valid"] = result["valid"]
    state["validation_confidence"] = result["confidence"]
    state["validation_issues"] = result["issues"]
    
    logger.info(
        f"✅ Validation: valid={result['valid']}, "
        f"confidence={result['confidence']:.2f}"
    )
    
    return state

# =========================================================================
# PUBLIC API
# =========================================================================

def process_query(
    self,
    query: str,
    user_id: str,
    filters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a query through the complete RAG pipeline.
    
    Args:
        query: User query
        user_id: User ID (for filtering documents)
        filters: Optional metadata filters
    
    Returns:
        Dictionary with:
            - answer: Final answer
            - citations: List of citations
            - confidence: Overall confidence
            - metadata: Processing metadata
            - stats: Performance statistics
    
    Example:
        result = orchestrator.process_query(
            query="What is AI?",
            user_id="user123"
        )
        print(result['answer'])
    """
    logger.info(f"{'=' * 80}")
    logger.info(f"Processing query: '{query}'")
    logger.info(f"User ID: {user_id}")
    logger.info(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Initialize state
    initial_state: RAGState = {
        "query": query,
        "user_id": user_id,
        "filters": filters or {"user_id": user_id},
        "language": "en",
        "language_confidence": 0.0,
        "query_type": "",
        "search_queries": [],
        "documents": [],
        "total_retrieved": 0,
        "analysis": "",
        "sources_used": [],
        "analysis_confidence": 0.0,
        "answer": "",
        "citations": [],
        "validation_valid": True,
        "validation_confidence": 0.0,
        "validation_issues": [],
        "processing_time": 0.0,
        "total_cost": 0.0,
        "agent_stats": {},
        "errors": []
    }
    
    try:
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        total_cost = self._calculate_total_cost()
        
        # Build response
        result = {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "confidence": self._calculate_overall_confidence(final_state),
            "language": final_state["language"],
            "query_type": final_state["query_type"],
            "sources": [
                {
                    "id": doc["id"],
                    "text": doc["text"][:200] + "...",
                    "score": doc["score"],
                    "metadata": doc["metadata"]
                }
                for doc in final_state["documents"]
            ],
            "metadata": {
                "processing_time": processing_time,
                "total_cost": total_cost,
                "documents_analyzed": len(final_state["documents"]),
                "validation_valid": final_state["validation_valid"],
                "validation_issues": final_state["validation_issues"]
            },
            "stats": self._get_all_stats()
        }
        
        logger.info(f"{'=' * 80}")
        logger.info(f"✅ Query processed successfully!")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Total cost: ${total_cost:.4f}")
        logger.info(f"{'=' * 80}")
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise AgentError("orchestrator", str(e))

def _calculate_total_cost(self) -> float:
    """Calculate total cost across all agents"""
    total = 0.0
    for agent in [self.router, self.planner, self.analyzer, self.synthesizer, self.validator]:
        total += agent.stats["total_cost"]
    return total

def _calculate_overall_confidence(self, state: RAGState) -> float:
    """
    Calculate overall confidence score.
    
    Combines:
    - Language detection confidence
    - Analysis confidence
    - Validation confidence
    """
    weights = {
        "language": 0.2,
        "analysis": 0.4,
        "validation": 0.4
    }
    
    confidence = (
        weights["language"] * state["language_confidence"] +
        weights["analysis"] * state["analysis_confidence"] +
        weights["validation"] * state["validation_confidence"]
    )
    
    return confidence

def _get_all_stats(self) -> Dict[str, Any]:
    """Get statistics from all agents"""
    return {
        "router": self.router.get_stats(),
        "planner": self.planner.get_stats(),
        "analyzer": self.analyzer.get_stats(),
        "synthesizer": self.synthesizer.get_stats(),
        "validator": self.validator.get_stats()
    }

def reset_stats(self):
    """Reset statistics for all agents"""
    for agent in [self.router, self.planner, self.analyzer, self.synthesizer, self.validator]:
        agent.reset_stats()
    logger.info("All agent stats reset")

#=============================================================================
#TESTING
#=============================================================================
    if name == "main":
    print("=" * 80)
    print("TESTING RAG ORCHESTRATOR")
    print("=" * 80)
# Note: This requires documents in vector store
# Run the vector store test first to add sample documents

    orchestrator = RAGOrchestrator()
    
    # Test query
    test_query = "What is the capital of India?"
    
    print(f"\n🚀 Processing query: '{test_query}'")
    print("=" * 80)
    
    result = orchestrator.process_query(
        query=test_query,
        user_id="test_user"
    )
    
    print(f"\n✅ FINAL RESULT:")
    print("=" * 80)
    print(f"\n📝 Answer:\n{result['answer']}")
    print(f"\n📊 Metadata:")
    print(f"  Language: {result['language']}")
    print(f"  Query type: {result['query_type']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Processing time: {result['metadata']['processing_time']:.2f}s")
    print(f"  Total cost: ${result['metadata']['total_cost']:.4f}")
    print(f"  Documents analyzed: {result['metadata']['documents_analyzed']}")
    print(f"  Validation valid: {result['metadata']['validation_valid']}")
    
    print(f"\n📚 Sources ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  {i}. Score: {source['score']:.4f}")
        print(f"     Text: {source['text'][:80]}...")
    
    print(f"\n📈 Agent Statistics:")
    for agent_name, stats in result['stats'].items():
        print(f"\n  {agent_name.upper()}:")
        print(f"    Calls: {stats['calls']}")
        print(f"    Tokens: {stats['total_tokens']}")
        print(f"    Cost: ${stats['total_cost']:.4f}")
        print(f"    Time: {stats['total_time']:.2f}s")
    
    print("\n" + "=" * 80)
    print("✅ RAG ORCHESTRATOR WORKING CORRECTLY!")
    print("=" * 80)
