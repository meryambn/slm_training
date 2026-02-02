from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time

class ToolCallRecord(BaseModel):
    step: int
    tool: str
    args: Dict[str, Any]
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0

class EpisodeTrace(BaseModel):
    episode_id: str
    task_id: str
    category: str
    question: str
    ground_truth: str
    final_answer: Optional[str] = None
    is_correct: bool = False
    status: str = "failed"
    step_count: int = 0
    duration_seconds: float = 0.0
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    failure_category: Optional[str] = None
    error_recovery_observed: bool = False
    consensus_agreement: Optional[float] = None
    semantic_entropy: Optional[float] = None
    sample_answers: List[str] = Field(default_factory=list)
    router_decision: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return self.model_dump_json()

class EvaluationSummary(BaseModel):
    total_tasks: int
    completed_tasks: int
    goal_completion_rate: float
    goal_completion_ci95: Tuple[float, float] = (0.0, 0.0)
    step_level_success_rate: float
    error_recovery_rate: float
    avg_steps_per_task: float
    avg_steps_se: float = 0.0
    avg_latency_seconds: float
    avg_latency_se: float = 0.0
    tool_distribution: Dict[str, int]
    failure_distribution: Dict[str, int]
    early_exit_count: int = 0
    tool_routed_count: int = 0
    mean_consensus_agreement: float = 0.0
    mean_semantic_entropy: float = 0.0
    mode: str = "baseline"
