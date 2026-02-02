"""
Failure Mode Taxonomy for Agent Evaluation.
Defines 6 formal failure categories and provides classification logic.
"""
from typing import Optional, List, Dict, Any
from evaluation.schemas import ToolCallRecord

FAILURE_CATEGORIES = {
    "WRONG_TOOL_SELECTION": "Chose an inappropriate tool for the current sub-problem (e.g. search for local file).",
    "MALFORMED_TOOL_CALL": "Missing required arguments, invalid schema, or empty execution payload.",
    "TOOL_EXECUTION_FAILURE": "Tool raised an unhandled error/timeout and the agent failed to self-correct.",
    "INFINITE_LOOP": "Repeated identical or oscillating tool calls without progress.",
    "PREMATURE_TERMINATION": "Agent terminated prematurely before fulfilling necessary requirements.",
    "HALLUCINATED_TOOL_OUTPUT": "Final answer contradicts or hallucinates data not present in tool observations."
}

def classify_failure(
    question: str,
    final_answer: Optional[str],
    tool_calls: List[ToolCallRecord],
    step_count: int,
    max_steps: int = 8
) -> str:
    """
    Classifies a failed episode into one of the 6 formal taxonomy categories.
    """
    # 1. Check for infinite loop: repeated identical tool + args
    seen_calls = set()
    for tc in tool_calls:
        call_sig = (tc.tool, tuple(sorted(tc.args.items())))
        if call_sig in seen_calls:
            return "INFINITE_LOOP"
        seen_calls.add(call_sig)

    # 2. Check for malformed tool calls
    for tc in tool_calls:
        if not tc.args or any(v is None or v == "" for v in tc.args.values()):
            return "MALFORMED_TOOL_CALL"

    # 3. Check for wrong tool selection
    q_lower = question.lower()
    for tc in tool_calls:
        if ("benchmarks/data" in q_lower or ".csv" in q_lower or ".json" in q_lower or ".txt" in q_lower) and tc.tool == "web_search":
            return "WRONG_TOOL_SELECTION"
        if ("prime" in q_lower or "mod " in q_lower or "determinant" in q_lower) and tc.tool == "file_io":
            return "WRONG_TOOL_SELECTION"

    # 4. Check for tool execution failure that wasn't recovered
    failed_calls = [tc for tc in tool_calls if not tc.success]
    if failed_calls:
        last_tc = tool_calls[-1]
        if not last_tc.success:
            return "TOOL_EXECUTION_FAILURE"

    # 5. Check for premature termination
    if not final_answer or final_answer.strip() == "":
        return "PREMATURE_TERMINATION"
    if step_count <= 1 and not tool_calls and any(k in q_lower for k in ["read", "compute", "count", "calculate", "find"]):
        return "PREMATURE_TERMINATION"

    # 6. Default fallback for incorrect answers despite tools executing
    return "HALLUCINATED_TOOL_OUTPUT"
