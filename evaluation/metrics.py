"""
Metrics computation module for Agent Evaluation Harness.
Evaluates:
- Goal Completion Rate with Wilson Score 95% Confidence Interval
- Step-Level Success Rate
- Error Recovery / Self-Correction Rate
- Tool Call Frequency & Distribution
- Failure Mode Distribution
- Latency & Step Standard Errors
- Mean Consensus Agreement & Semantic Entropy
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import statistics
from evaluation.schemas import EpisodeTrace, EvaluationSummary

def normalize_answer(s: Optional[str]) -> str:
    """Normalize answer string for robust comparison against ground truth."""
    if s is None:
        return ""
    s = str(s).strip()
    # Remove common prefix markers like "The answer is", "The result is:", "FINAL ANSWER:"
    s = re.sub(r"^(the\s+(?:answer|result)\s+is:?|final\s+answer:?|result:?)\s*", "", s, flags=re.IGNORECASE)
    # Remove currency symbols and commas in numbers (e.g. $16,139.73 -> 16139.73)
    s = s.replace("$", "").replace(",", "").replace("%", "")
    # Remove trailing period
    s = s.rstrip(".").strip()
    return s.lower()

def is_answer_correct(ground_truth: str, predicted: Optional[str]) -> bool:
    """Check whether predicted answer matches ground truth (numerical or string)."""
    if predicted is None or not str(predicted).strip():
        return False
    
    norm_gt = normalize_answer(ground_truth)
    norm_pred = normalize_answer(predicted)
    
    if norm_gt == norm_pred:
        return True

    # Try numerical comparison
    try:
        gt_num = float(norm_gt)
        # Extract potential number from predicted text
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", norm_pred)
        for num_str in nums:
            try:
                if abs(float(num_str) - gt_num) < 1e-2:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    # Substring containment for named entities (e.g. "Canberra" in "The capital is Canberra")
    if len(norm_gt) > 2 and norm_gt in norm_pred:
        return True

    return False

def compute_wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """
    Computes Wilson score 95% confidence interval for a binomial proportion.
    Returns (lower_bound_pct, upper_bound_pct).
    """
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    z2 = z ** 2
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / total) + (z2 / (4 * total ** 2)))) / denom
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return (round(lower, 2), round(upper, 2))

def compute_run_metrics(traces: List[EpisodeTrace], mode: str = "baseline") -> EvaluationSummary:
    total = len(traces)
    if total == 0:
        return EvaluationSummary(
            total_tasks=0,
            completed_tasks=0,
            goal_completion_rate=0.0,
            goal_completion_ci95=(0.0, 0.0),
            step_level_success_rate=0.0,
            error_recovery_rate=0.0,
            avg_steps_per_task=0.0,
            avg_steps_se=0.0,
            avg_latency_seconds=0.0,
            avg_latency_se=0.0,
            tool_distribution={},
            failure_distribution={},
            early_exit_count=0,
            tool_routed_count=0,
            mean_consensus_agreement=0.0,
            mean_semantic_entropy=0.0,
            mode=mode
        )

    completed = sum(1 for t in traces if t.is_correct)
    goal_completion_rate = (completed / total) * 100.0
    ci95 = compute_wilson_score_interval(completed, total)

    # Step-level success and tools
    total_tool_steps = 0
    successful_tool_steps = 0
    tool_dist: Dict[str, int] = {}
    
    # Error recovery
    episodes_with_tool_errors = 0
    recovered_episodes = 0

    step_counts = [t.step_count for t in traces]
    latencies = [t.duration_seconds for t in traces]
    failure_dist: Dict[str, int] = {}

    agreements = [t.consensus_agreement for t in traces if t.consensus_agreement is not None]
    entropies = [t.semantic_entropy for t in traces if t.semantic_entropy is not None]
    early_exits = sum(1 for t in traces if t.router_decision and t.router_decision.get("action") == "early_exit")
    tool_routes = sum(1 for t in traces if t.router_decision and t.router_decision.get("action") == "route_tool")

    for t in traces:
        had_error = False
        recovered = False
        
        for i, tc in enumerate(t.tool_calls):
            total_tool_steps += 1
            tool_dist[tc.tool] = tool_dist.get(tc.tool, 0) + 1
            if tc.success:
                successful_tool_steps += 1
            else:
                had_error = True
                if any(sub_tc.success for sub_tc in t.tool_calls[i+1:]) or t.is_correct:
                    recovered = True
        
        if had_error:
            episodes_with_tool_errors += 1
            if recovered:
                recovered_episodes += 1

        if not t.is_correct and t.failure_category:
            failure_dist[t.failure_category] = failure_dist.get(t.failure_category, 0) + 1

    step_success_rate = (successful_tool_steps / total_tool_steps * 100.0) if total_tool_steps > 0 else 100.0
    error_recovery_rate = (recovered_episodes / episodes_with_tool_errors * 100.0) if episodes_with_tool_errors > 0 else 100.0

    avg_steps = statistics.mean(step_counts) if step_counts else 0.0
    se_steps = (statistics.stdev(step_counts) / math.sqrt(total)) if len(step_counts) > 1 else 0.0

    avg_lat = statistics.mean(latencies) if latencies else 0.0
    se_lat = (statistics.stdev(latencies) / math.sqrt(total)) if len(latencies) > 1 else 0.0

    mean_agr = statistics.mean(agreements) if agreements else 0.0
    mean_ent = statistics.mean(entropies) if entropies else 0.0

    return EvaluationSummary(
        total_tasks=total,
        completed_tasks=completed,
        goal_completion_rate=round(goal_completion_rate, 2),
        goal_completion_ci95=ci95,
        step_level_success_rate=round(step_success_rate, 2),
        error_recovery_rate=round(error_recovery_rate, 2),
        avg_steps_per_task=round(avg_steps, 2),
        avg_steps_se=round(se_steps, 2),
        avg_latency_seconds=round(avg_lat, 2),
        avg_latency_se=round(se_lat, 2),
        tool_distribution=tool_dist,
        failure_distribution=failure_dist,
        early_exit_count=early_exits,
        tool_routed_count=tool_routes,
        mean_consensus_agreement=round(mean_agr, 3),
        mean_semantic_entropy=round(mean_ent, 3),
        mode=mode
    )
