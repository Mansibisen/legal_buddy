"""
LangGraph State Machine for Self-Correction Loop
Orchestrates grader, rewriter, and hallucination checker
"""
import logging
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from dataclasses import dataclass, field

try:
    from langgraph.graph import StateGraph
    from langgraph.graph.state import START, END
except ImportError:
    logging.warning("LangGraph not installed. Install with: pip install langgraph")


logger = logging.getLogger(__name__)


# State type definitions
class WorkflowState(Enum):
    """States in the self-correction workflow"""
    RETRIEVE = "retrieve"         # Initial retrieval
    GRADE = "grade"               # Grade retrieved documents
    REWRITE = "rewrite"           # Rewrite query if needed
    RETRIEVE_AGAIN = "retrieve_again"  # Retrieve with new query
    GENERATE = "generate"         # Generate answer
    CHECK_HALLUCINATION = "check_hallucination"  # Validate answer
    REFINE = "refine"             # Refine if hallucination found
    COMPLETE = "complete"         # Workflow complete


@dataclass
class SelfCorrectionWorkflowState:
    """State for self-correction workflow"""
    question: str
    original_query: str
    current_query: str
    
    # Retrieval state
    retrieved_documents: List[str] = field(default_factory=list)
    retrieved_metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    # Grading state
    grade_assessment: Dict[str, Any] = field(default_factory=dict)
    relevant_documents: List[str] = field(default_factory=list)
    relevant_count: int = 0
    
    # Rewriting state
    rewrite_history: List[Dict[str, Any]] = field(default_factory=list)
    current_rewrite: Optional[Dict[str, Any]] = None
    
    # Answer generation
    answer: Optional[str] = None
    answer_context: Optional[str] = None
    
    # Hallucination check
    hallucination_assessment: Dict[str, Any] = field(default_factory=dict)
    is_hallucinating: bool = False
    
    # Workflow control
    attempt_count: int = 0
    max_attempts: int = 3
    current_state: str = WorkflowState.RETRIEVE.value
    workflow_complete: bool = False
    success: bool = False
    
    # Metrics
    retrieval_time: float = 0.0
    grading_time: float = 0.0
    rewriting_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0


class SelfCorrectionOrchestrator:
    """
    Orchestrates the complete self-correction workflow using LangGraph
    """
    
    def __init__(self):
        """Initialize orchestrator"""
        self.graph = None
        self._setup_graph()
    
    def _setup_graph(self):
        """Setup LangGraph state machine"""
        try:
            from langgraph.graph import StateGraph
            from langgraph.graph.state import START, END
            
            # Create state graph
            workflow = StateGraph(SelfCorrectionWorkflowState)
            
            # Add nodes
            workflow.add_node("retrieve", self._node_retrieve)
            workflow.add_node("grade", self._node_grade)
            workflow.add_node("rewrite", self._node_rewrite)
            workflow.add_node("retrieve_again", self._node_retrieve_again)
            workflow.add_node("generate", self._node_generate)
            workflow.add_node("check_hallucination", self._node_check_hallucination)
            workflow.add_node("refine", self._node_refine)
            workflow.add_node("complete", self._node_complete)
            
            # Add edges
            workflow.add_edge(START, "retrieve")
            workflow.add_conditional_edges(
                "grade",
                self._decide_grade,
                {
                    "continue": "generate",
                    "rewrite": "rewrite"
                }
            )
            workflow.add_edge("rewrite", "retrieve_again")
            workflow.add_edge("retrieve_again", "grade")
            workflow.add_edge("generate", "check_hallucination")
            workflow.add_conditional_edges(
                "check_hallucination",
                self._decide_hallucination,
                {
                    "complete": "complete",
                    "refine": "refine"
                }
            )
            workflow.add_edge("refine", "complete")
            workflow.add_edge("complete", END)
            
            self.graph = workflow.compile()
            logger.info("LangGraph state machine compiled successfully")
            
        except ImportError:
            logger.warning("LangGraph not available. Fallback to manual orchestration.")
            self.graph = None
    
    def execute_workflow(
        self,
        question: str,
        retriever_fn,
        grader_fn,
        rewriter_fn,
        generator_fn,
        hallucination_checker_fn,
        initial_documents: Optional[List[str]] = None
    ) -> SelfCorrectionWorkflowState:
        """
        Execute complete self-correction workflow
        
        Args:
            question: Question to answer
            retriever_fn: Function to retrieve documents
            grader_fn: Function to grade documents
            rewriter_fn: Function to rewrite query
            generator_fn: Function to generate answer
            hallucination_checker_fn: Function to check hallucinations
            initial_documents: Optional initial documents
            
        Returns:
            Final workflow state
        """
        import time
        
        # Initialize state
        state = SelfCorrectionWorkflowState(
            question=question,
            original_query=question,
            current_query=question
        )
        
        start_time = time.time()
        
        try:
            if self.graph:
                # Use LangGraph execution
                logger.info("Executing workflow via LangGraph")
                final_state = self.graph.invoke(
                    state,
                    config={
                        "retriever_fn": retriever_fn,
                        "grader_fn": grader_fn,
                        "rewriter_fn": rewriter_fn,
                        "generator_fn": generator_fn,
                        "hallucination_checker_fn": hallucination_checker_fn
                    }
                )
            else:
                # Fallback manual orchestration
                logger.info("Executing workflow via manual orchestration")
                final_state = self._execute_manual_workflow(
                    state,
                    retriever_fn,
                    grader_fn,
                    rewriter_fn,
                    generator_fn,
                    hallucination_checker_fn,
                    initial_documents
                )
            
            final_state.total_time = time.time() - start_time
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            state.workflow_complete = True
            state.success = False
            state.total_time = time.time() - start_time
            return state
    
    def _execute_manual_workflow(
        self,
        state: SelfCorrectionWorkflowState,
        retriever_fn,
        grader_fn,
        rewriter_fn,
        generator_fn,
        hallucination_checker_fn,
        initial_documents: Optional[List[str]] = None
    ) -> SelfCorrectionWorkflowState:
        """Fallback manual workflow execution"""
        import time
        
        # Step 1: Retrieve documents
        logger.info(f"Step 1: Retrieving documents for: {state.question}")
        start = time.time()
        if initial_documents:
            state.retrieved_documents = initial_documents
        else:
            state.retrieved_documents = retriever_fn(state.current_query) or []
        state.retrieval_time = time.time() - start
        
        # Retry loop
        max_retries = state.max_attempts
        for attempt in range(max_retries):
            state.attempt_count = attempt + 1
            
            # Step 2: Grade documents
            logger.info(f"Attempt {state.attempt_count}: Grading documents")
            start = time.time()
            state.grade_assessment = grader_fn(state.question, state.retrieved_documents) or {}
            state.grading_time = time.time() - start
            state.relevant_count = state.grade_assessment.get("relevant_count", 0)
            state.relevant_documents = state.grade_assessment.get("relevant_documents", [])
            
            # Check if we have enough relevant documents
            if state.relevant_count > 0:
                logger.info(f"Found {state.relevant_count} relevant documents")
                break
            
            # Step 3: Rewrite query if documents not relevant
            if attempt < max_retries - 1:
                logger.info(f"Rewriting query (attempt {state.attempt_count}/{max_retries})")
                start = time.time()
                rewrite_result = rewriter_fn(
                    state.current_query,
                    f"No relevant documents found. Try: {state.grade_assessment.get('feedback', '')}"
                ) or {}
                state.rewriting_time = time.time() - start
                
                state.current_query = rewrite_result.get("rewritten_query", state.current_query)
                state.rewrite_history.append(rewrite_result)
                
                # Step 4: Retrieve with new query
                logger.info(f"Retrieving with rewritten query: {state.current_query}")
                start = time.time()
                state.retrieved_documents = retriever_fn(state.current_query) or []
                state.retrieval_time = time.time() - start
        
        # Step 5: Generate answer
        logger.info("Generating answer")
        start = time.time()
        state.answer = generator_fn(
            state.question,
            state.relevant_documents
        ) or ""
        state.generation_time = time.time() - start
        
        # Step 6: Check for hallucinations
        logger.info("Checking answer for hallucinations")
        state.hallucination_assessment = hallucination_checker_fn(
            state.question,
            state.answer,
            state.relevant_documents
        ) or {}
        state.is_hallucinating = state.hallucination_assessment.get("is_hallucinating", False)
        
        # Step 7: Finalize
        state.workflow_complete = True
        state.success = not state.is_hallucinating and state.relevant_count > 0
        
        return state
    
    # Node definitions (for LangGraph)
    def _node_retrieve(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Retrieve documents"""
        state.current_state = WorkflowState.RETRIEVE.value
        return state
    
    def _node_grade(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Grade retrieved documents"""
        state.current_state = WorkflowState.GRADE.value
        return state
    
    def _node_rewrite(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Rewrite query"""
        state.current_state = WorkflowState.REWRITE.value
        return state
    
    def _node_retrieve_again(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Retrieve with rewritten query"""
        state.current_state = WorkflowState.RETRIEVE_AGAIN.value
        return state
    
    def _node_generate(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Generate answer"""
        state.current_state = WorkflowState.GENERATE.value
        return state
    
    def _node_check_hallucination(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Check for hallucinations"""
        state.current_state = WorkflowState.CHECK_HALLUCINATION.value
        return state
    
    def _node_refine(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Refine answer if hallucination detected"""
        state.current_state = WorkflowState.REFINE.value
        return state
    
    def _node_complete(self, state: SelfCorrectionWorkflowState) -> SelfCorrectionWorkflowState:
        """Complete workflow"""
        state.current_state = WorkflowState.COMPLETE.value
        state.workflow_complete = True
        return state
    
    # Decision functions
    def _decide_grade(self, state: SelfCorrectionWorkflowState) -> Literal["continue", "rewrite"]:
        """Decide whether to continue or rewrite"""
        if state.relevant_count > 0:
            return "continue"
        elif state.attempt_count < state.max_attempts:
            return "rewrite"
        else:
            return "continue"  # Continue anyway if max attempts reached
    
    def _decide_hallucination(self, state: SelfCorrectionWorkflowState) -> Literal["complete", "refine"]:
        """Decide whether to complete or refine"""
        if state.is_hallucinating:
            return "refine"
        else:
            return "complete"


# Convenience function
def run_self_correcting_workflow(
    question: str,
    retriever_fn,
    grader_fn,
    rewriter_fn,
    generator_fn,
    hallucination_checker_fn,
    max_attempts: int = 3
) -> SelfCorrectionWorkflowState:
    """
    Run complete self-correcting workflow
    
    Args:
        question: Question to answer
        retriever_fn: Function to retrieve documents
        grader_fn: Function to grade documents
        rewriter_fn: Function to rewrite query
        generator_fn: Function to generate answer
        hallucination_checker_fn: Function to check hallucinations
        max_attempts: Maximum retrieval attempts
        
    Returns:
        Final workflow state with results
    """
    orchestrator = SelfCorrectionOrchestrator()
    return orchestrator.execute_workflow(
        question,
        retriever_fn,
        grader_fn,
        rewriter_fn,
        generator_fn,
        hallucination_checker_fn
    )
