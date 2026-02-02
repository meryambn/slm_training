from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    """
    Strictly typed agent state for LangGraph ReAct graph execution.
    Utilizes add_messages reducer for conversational message history
    and operator.add for incremental tool call history.
    """
    task_id: str
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int
    max_steps: int
    tool_call_history: Annotated[List[Dict[str, Any]], operator.add]
    final_answer: Optional[str]
    status: str
    confidence_score: float
    consensus_agreement: float
    semantic_entropy: float
    sample_answers: List[str]
    router_decision: Optional[Dict[str, Any]]
    execution_start_time: float
    execution_total_time: float
