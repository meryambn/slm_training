"""
Utility to compare two evaluation JSONL trace runs and generate comparative metrics.
"""
import sys
import json
from pathlib import Path
from tabulate import tabulate
from evaluation.schemas import EpisodeTrace
from evaluation.metrics import compute_run_metrics

def compare_traces(baseline_path: str, routed_path: str):
    base_traces = []
    with open(baseline_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_traces.append(EpisodeTrace.model_validate_json(line))

    routed_traces = []
    with open(routed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                routed_traces.append(EpisodeTrace.model_validate_json(line))

    base_sum = compute_run_metrics(base_traces, mode="baseline")
    routed_sum = compute_run_metrics(routed_traces, mode="uncertainty_routed")

    base_tools = sum(base_sum.tool_distribution.values())
    routed_tools = sum(routed_sum.tool_distribution.values())
    tool_diff = ((base_tools - routed_tools) / base_tools * 100) if base_tools > 0 else 0
    lat_diff = ((base_sum.avg_latency_seconds - routed_sum.avg_latency_seconds) / base_sum.avg_latency_seconds * 100) if base_sum.avg_latency_seconds > 0 else 0

    table = [
        ["Total Tasks Evaluated", base_sum.total_tasks, routed_sum.total_tasks, "-"],
        ["Tasks Solved (Exact Match)", base_sum.completed_tasks, routed_sum.completed_tasks, f"{routed_sum.completed_tasks - base_sum.completed_tasks:+d}"],
        ["Goal Completion Rate", f"{base_sum.goal_completion_rate}%", f"{routed_sum.goal_completion_rate}%", f"{routed_sum.goal_completion_rate - base_sum.goal_completion_rate:+.2f}%"],
        ["Total Tool Invocations", base_tools, routed_tools, f"-{tool_diff:.1f}%"],
        ["Avg Steps per Task", base_sum.avg_steps_per_task, routed_sum.avg_steps_per_task, f"{routed_sum.avg_steps_per_task - base_sum.avg_steps_per_task:+.2f}"],
        ["Avg Latency per Task", f"{base_sum.avg_latency_seconds}s", f"{routed_sum.avg_latency_seconds}s", f"-{lat_diff:.1f}% speedup"],
        ["Step-Level Success Rate", f"{base_sum.step_level_success_rate}%", f"{routed_sum.step_level_success_rate}%", f"{routed_sum.step_level_success_rate - base_sum.step_level_success_rate:+.2f}%"],
        ["Error Recovery Rate", f"{base_sum.error_recovery_rate}%", f"{routed_sum.error_recovery_rate}%", f"{routed_sum.error_recovery_rate - base_sum.error_recovery_rate:+.2f}%"],
        ["Tool Distribution", str(base_sum.tool_distribution), str(routed_sum.tool_distribution), "-"],
        ["Failure Breakdown", str(base_sum.failure_distribution), str(routed_sum.failure_distribution), "-"]
    ]
    headers = ["Metric", "Baseline ReAct", "Uncertainty-Routed", "Delta / Impact"]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid") + "\n")
    return base_sum, routed_sum

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Find latest baseline and routed
        traces_dir = Path("evaluation/traces")
        baselines = sorted(traces_dir.glob("gaia_baseline_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        routeds = sorted(traces_dir.glob("gaia_uncertainty_routed_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if baselines and routeds:
            compare_traces(str(baselines[0]), str(routeds[0]))
        else:
            print("Usage: python scripts/compare_runs.py <baseline_trace.jsonl> <routed_trace.jsonl>")
    else:
        compare_traces(sys.argv[1], sys.argv[2])
