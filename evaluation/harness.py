"""
Agent-Agnostic Evaluation Harness with Multi-Arm Benchmarking and Statistical CIs.
Executes tasks, logs structured JSONL traces, computes metrics with Wilson 95% CIs,
and renders multi-arm comparative benchmark tables.
"""
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from tabulate import tabulate

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluation.schemas import EpisodeTrace, ToolCallRecord, EvaluationSummary
from evaluation.metrics import is_answer_correct, compute_run_metrics, compute_wilson_score_interval
from evaluation.taxonomy import classify_failure
from benchmarks.gaia_loader import load_gaia_subset, GAIATask
from benchmarks.real_benchmarks import load_gsm8k_benchmark
from agent.react_graph import ReActAgent

TRACES_DIR = Path("evaluation/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)

class EvaluationHarness:
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or TRACES_DIR

    def run_benchmark(
        self,
        agent_runner: Callable[[str, str, str], Dict[str, Any]],
        tasks: List[GAIATask],
        run_name: str = "gaia_baseline",
        mode: str = "baseline"
    ) -> EvaluationSummary:
        trace_file = self.log_dir / f"{run_name}_{int(time.time())}.jsonl"
        traces: List[EpisodeTrace] = []

        print(f"\n=======================================================", flush=True)
        print(f" Starting Evaluation Run: {run_name} (Arm: {mode})", flush=True)
        print(f" Total Tasks: {len(tasks)}", flush=True)
        print(f" Log Destination: {trace_file}", flush=True)
        print(f"=======================================================\n", flush=True)

        with open(trace_file, "w", encoding="utf-8") as f_log:
            for idx, task in enumerate(tasks, 1):
                start_t = time.perf_counter()
                print(f"[{idx}/{len(tasks)}] Task {task.task_id} ({task.category}): {task.question[:65]}...", flush=True)

                try:
                    agent_res = agent_runner(task.question, task.task_id, mode)
                    duration = time.perf_counter() - start_t
                    
                    final_ans = agent_res.get("final_answer")
                    is_correct = is_answer_correct(task.ground_truth, final_ans)
                    
                    raw_tc = agent_res.get("tool_call_history", [])
                    tool_calls = [
                        ToolCallRecord(
                            step=item.get("step", 0),
                            tool=item.get("tool", "unknown"),
                            args=item.get("args", {}),
                            success=item.get("result", {}).get("success", True),
                            output=str(item.get("result", {}).get("output", ""))[:500],
                            error=item.get("result", {}).get("error"),
                            execution_time=item.get("result", {}).get("execution_time", 0.0)
                        )
                        for item in raw_tc
                    ]

                    step_count = agent_res.get("step_count", len(tool_calls) + 1)
                    
                    failure_cat = None
                    if not is_correct:
                        failure_cat = classify_failure(
                            question=task.question,
                            final_answer=final_ans,
                            tool_calls=tool_calls,
                            step_count=step_count
                        )

                    had_error = any(not tc.success for tc in tool_calls)
                    error_recovered = False
                    if had_error:
                        for i, tc in enumerate(tool_calls):
                            if not tc.success and (any(sub.success for sub in tool_calls[i+1:]) or is_correct):
                                error_recovered = True
                                break

                    trace = EpisodeTrace(
                        episode_id=f"{run_name}_{task.task_id}",
                        task_id=task.task_id,
                        category=task.category,
                        question=task.question,
                        ground_truth=task.ground_truth,
                        final_answer=final_ans,
                        is_correct=is_correct,
                        status="success" if is_correct else "failed",
                        step_count=step_count,
                        duration_seconds=round(duration, 2),
                        tool_calls=tool_calls,
                        failure_category=failure_cat,
                        error_recovery_observed=error_recovered,
                        consensus_agreement=agent_res.get("consensus_agreement"),
                        semantic_entropy=agent_res.get("semantic_entropy"),
                        sample_answers=agent_res.get("sample_answers", []),
                        router_decision=agent_res.get("router_decision")
                    )

                except Exception as e:
                    duration = time.perf_counter() - start_t
                    trace = EpisodeTrace(
                        episode_id=f"{run_name}_{task.task_id}",
                        task_id=task.task_id,
                        category=task.category,
                        question=task.question,
                        ground_truth=task.ground_truth,
                        final_answer=None,
                        is_correct=False,
                        status="crash",
                        step_count=0,
                        duration_seconds=round(duration, 2),
                        tool_calls=[],
                        failure_category="TOOL_EXECUTION_FAILURE"
                    )

                traces.append(trace)
                f_log.write(trace.to_jsonl() + "\n")
                f_log.flush()

                status_flag = "PASS [OK]" if trace.is_correct else f"FAIL [{trace.failure_category}]"
                print(f"      -> Ans: '{trace.final_answer}' | GT: '{task.ground_truth}' | Result: {status_flag} ({trace.duration_seconds}s)", flush=True)

        summary = compute_run_metrics(traces, mode=mode)
        self.render_summary(summary)
        return summary

    def render_summary(self, summary: EvaluationSummary):
        headers = ["Metric", "Value"]
        ci_str = f"[{summary.goal_completion_ci95[0]}%, {summary.goal_completion_ci95[1]}%]"
        table = [
            ["Evaluation Arm", summary.mode],
            ["Total Tasks", summary.total_tasks],
            ["Completed (Correct)", summary.completed_tasks],
            ["Goal Completion Rate", f"{summary.goal_completion_rate}% (95% CI: {ci_str})"],
            ["Step-Level Success Rate", f"{summary.step_level_success_rate}%"],
            ["Error Recovery Rate", f"{summary.error_recovery_rate}%"],
            ["Avg Steps / Task", f"{summary.avg_steps_per_task} ± {summary.avg_steps_se}"],
            ["Avg Latency / Task", f"{summary.avg_latency_seconds}s ± {summary.avg_latency_se}s"],
            ["Early-Exit Decisions", summary.early_exit_count],
            ["Tool-Routed Decisions", summary.tool_routed_count],
            ["Mean Consensus Agreement", summary.mean_consensus_agreement],
            ["Mean Semantic Entropy", summary.mean_semantic_entropy],
            ["Tool Calls Total", sum(summary.tool_distribution.values())],
            ["Tool Distribution", str(summary.tool_distribution)],
            ["Failure Taxonomy Breakdown", str(summary.failure_distribution)]
        ]
        print("\n" + tabulate(table, headers=headers, tablefmt="grid") + "\n", flush=True)

def render_multi_arm_table(summaries: Dict[str, EvaluationSummary]):
    headers = ["Metric", "Zero-Shot Direct", "Baseline ReAct", "Uncertainty-Routed (UQ)", "Impact vs Baseline"]
    zs = summaries.get("zero_shot")
    base = summaries.get("baseline")
    routed = summaries.get("routed")

    if not base or not routed:
        return

    delta_acc = routed.goal_completion_rate - base.goal_completion_rate
    base_tools = sum(base.tool_distribution.values())
    routed_tools = sum(routed.tool_distribution.values())
    tool_diff = ((base_tools - routed_tools) / base_tools * 100) if base_tools > 0 else 0
    lat_diff = ((base.avg_latency_seconds - routed.avg_latency_seconds) / base.avg_latency_seconds * 100) if base.avg_latency_seconds > 0 else 0

    zs_acc = f"{zs.goal_completion_rate}%" if zs else "N/A"
    zs_tools = sum(zs.tool_distribution.values()) if zs else 0
    zs_steps = f"{zs.avg_steps_per_task} ± {zs.avg_steps_se}" if zs else "1.0 ± 0.0"
    zs_lat = f"{zs.avg_latency_seconds}s ± {zs.avg_latency_se}s" if zs else "N/A"

    table = [
        ["Goal Completion Rate", zs_acc, f"{base.goal_completion_rate}%", f"{routed.goal_completion_rate}%", f"{delta_acc:+.2f}%"],
        ["Wilson 95% CI", f"[{zs.goal_completion_ci95[0]}%, {zs.goal_completion_ci95[1]}%]" if zs else "N/A", f"[{base.goal_completion_ci95[0]}%, {base.goal_completion_ci95[1]}%]", f"[{routed.goal_completion_ci95[0]}%, {routed.goal_completion_ci95[1]}%]", "-"],
        ["Total Tool Calls", zs_tools, base_tools, routed_tools, f"-{tool_diff:.1f}% reduction"],
        ["Avg Steps per Task", zs_steps, f"{base.avg_steps_per_task} ± {base.avg_steps_se}", f"{routed.avg_steps_per_task} ± {routed.avg_steps_se}", f"{routed.avg_steps_per_task - base.avg_steps_per_task:+.2f}"],
        ["Avg Latency per Task", zs_lat, f"{base.avg_latency_seconds}s ± {base.avg_latency_se}s", f"{routed.avg_latency_seconds}s ± {routed.avg_latency_se}s", f"-{lat_diff:.1f}% speedup"],
        ["Step-Level Success Rate", "100.0%" if zs else "N/A", f"{base.step_level_success_rate}%", f"{routed.step_level_success_rate}%", f"{routed.step_level_success_rate - base.step_level_success_rate:+.2f}%"],
        ["Error Recovery Rate", "N/A", f"{base.error_recovery_rate}%", f"{routed.error_recovery_rate}%", f"{routed.error_recovery_rate - base.error_recovery_rate:+.2f}%"],
        ["Early-Exit Count", "All" if zs else "N/A", "0", routed.early_exit_count, f"+{routed.early_exit_count}"],
        ["Tool-Routed Count", "0", base.total_tasks, routed.tool_routed_count, f"{routed.tool_routed_count - base.total_tasks:+d}"]
    ]
    print("\n=======================================================================================", flush=True)
    print("                      RIGOROUS MULTI-ARM BENCHMARK COMPARISON TABLE                     ", flush=True)
    print("=======================================================================================", flush=True)
    print(tabulate(table, headers=headers, tablefmt="grid"), flush=True)

def run_evaluation_cli():
    parser = argparse.ArgumentParser(description="Run Agent Evaluation Harness on Benchmark Datasets")
    parser.add_argument("--dataset", type=str, default="gaia", choices=["gaia", "gsm8k"], help="Benchmark dataset to evaluate (gaia or gsm8k)")
    parser.add_argument("--subset-size", type=int, default=15, help="Number of benchmark tasks to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Filter tasks by category (for GAIA)")
    parser.add_argument("--mode", type=str, default="routed", choices=["baseline", "zero_shot", "routed"], help="Single arm execution mode")
    parser.add_argument("--multi-arm", action="store_true", help="Execute full 3-arm benchmark (Zero-Shot vs Baseline vs Routed)")
    args = parser.parse_args()

    agent = ReActAgent()
    harness = EvaluationHarness()
    
    if args.dataset == "gsm8k":
        tasks = load_gsm8k_benchmark(split="test", limit=args.subset_size)
    else:
        tasks = load_gaia_subset(category=args.category)[:args.subset_size]

    dataset_tag = args.dataset

    if args.multi_arm:
        summaries = {}
        # Arm 1: Zero-Shot Direct (Lower Baseline)
        summaries["zero_shot"] = harness.run_benchmark(
            agent_runner=lambda q, tid, m: agent.run(q, tid, mode=m),
            tasks=tasks,
            run_name=f"{dataset_tag}_zero_shot",
            mode="zero_shot"
        )
        # Arm 2: Baseline ReAct
        summaries["baseline"] = harness.run_benchmark(
            agent_runner=lambda q, tid, m: agent.run(q, tid, mode=m),
            tasks=tasks,
            run_name=f"{dataset_tag}_baseline",
            mode="baseline"
        )
        # Arm 3: Uncertainty-Routed ReAct
        summaries["routed"] = harness.run_benchmark(
            agent_runner=lambda q, tid, m: agent.run(q, tid, mode=m),
            tasks=tasks,
            run_name=f"{dataset_tag}_uncertainty_routed",
            mode="routed"
        )
        render_multi_arm_table(summaries)
    else:
        harness.run_benchmark(
            agent_runner=lambda q, tid, m: agent.run(q, tid, mode=m),
            tasks=tasks,
            run_name=f"{dataset_tag}_{args.mode}",
            mode=args.mode
        )

if __name__ == "__main__":
    run_evaluation_cli()
