"""
FastAPI Server for SLM ReAct Agent & Evaluation Harness.
Provides RESTful endpoints for interactive agent querying,
real-time tool execution, and benchmark metrics reporting.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import json
import time
import httpx

from agent.react_graph import ReActAgent

app = FastAPI(
    title="SLM ReAct Agent API",
    version="2.0.0",
    description="REST API for Qwen2.5 ReAct Agent with Uncertainty Quantification and Multi-Arm Telemetry"
)

# Global agent singleton
_agent_instance: Optional[ReActAgent] = None

def get_agent() -> ReActAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ReActAgent()
    return _agent_instance

class SolveRequest(BaseModel):
    question: str = Field(..., description="The natural language task or problem to solve")
    task_id: str = Field(default="interactive_task", description="Identifier for this task")
    mode: Literal["baseline", "zero_shot", "routed"] = Field(
        default="routed",
        description="Evaluation arm: zero_shot (direct answer), baseline (standard ReAct), or routed (uncertainty-aware)"
    )
    max_steps: int = Field(default=8, description="Maximum ReAct reasoning steps")

class SolveResponse(BaseModel):
    task_id: str
    question: str
    mode: str
    final_answer: Optional[str]
    status: str
    step_count: int
    duration_seconds: float
    tool_calls: List[Dict[str, Any]]
    consensus_agreement: Optional[float] = None
    semantic_entropy: Optional[float] = None
    sample_answers: List[str] = Field(default_factory=list)
    router_decision: Optional[Dict[str, Any]] = None

@app.get("/health")
def health_check():
    ollama_ok = False
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            ollama_ok = (resp.status_code == 200)
    except Exception:
        ollama_ok = False

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama_available": ollama_ok,
        "model": "qwen2.5:7b-instruct",
        "timestamp": time.time()
    }

@app.post("/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest):
    agent = get_agent()
    try:
        res = agent.run(
            question=req.question,
            task_id=req.task_id,
            mode=req.mode
        )
        return SolveResponse(
            task_id=res.get("task_id", req.task_id),
            question=req.question,
            mode=req.mode,
            final_answer=res.get("final_answer"),
            status=res.get("status", "unknown"),
            step_count=res.get("step_count", 0),
            duration_seconds=round(res.get("execution_total_time", 0.0), 3),
            tool_calls=res.get("tool_call_history", []),
            consensus_agreement=res.get("consensus_agreement"),
            semantic_entropy=res.get("semantic_entropy"),
            sample_answers=res.get("sample_answers", []),
            router_decision=res.get("router_decision")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/traces/latest")
def get_latest_traces(limit: int = 10):
    traces_dir = Path("evaluation/traces")
    if not traces_dir.exists():
        return {"traces": []}
    
    files = sorted(traces_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {"traces": []}
    
    latest_file = files[0]
    traces = []
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
                if len(traces) >= limit:
                    break
    return {
        "file": latest_file.name,
        "count": len(traces),
        "traces": traces
    }
