import pytest
from evaluation.metrics import normalize_answer, is_answer_correct, compute_run_metrics
from evaluation.taxonomy import classify_failure
from evaluation.schemas import EpisodeTrace, ToolCallRecord

def test_normalize_answer():
    assert normalize_answer("$12,500.00") == "12500.00"
    assert normalize_answer("Final Answer: Canberra.") == "canberra"
    assert normalize_answer("  1082 % ") == "1082"
    assert normalize_answer("The result is: 42") == "42"

def test_is_answer_correct():
    assert is_answer_correct("1082", "1082") is True
    assert is_answer_correct("16139.73", "$16,139.73") is True
    assert is_answer_correct("-0.985", "-0.98501") is True
    assert is_answer_correct("Canberra", "The capital of Australia is Canberra.") is True
    assert is_answer_correct("Mary Shelley", "Written by Mary Shelley") is True
    assert is_answer_correct("100", "200") is False
    assert is_answer_correct("Paris", "London") is False
    assert is_answer_correct("42", None) is False

def test_taxonomy_infinite_loop():
    tool_calls = [
        ToolCallRecord(step=1, tool="python_repl", args={"code": "1+1"}, success=True, output="2"),
        ToolCallRecord(step=2, tool="python_repl", args={"code": "1+1"}, success=True, output="2")
    ]
    failure = classify_failure(
        question="Compute something",
        final_answer=None,
        tool_calls=tool_calls,
        step_count=3
    )
    assert failure == "INFINITE_LOOP"

def test_taxonomy_malformed_tool_call():
    tool_calls = [
        ToolCallRecord(step=1, tool="python_repl", args={"code": ""}, success=False, output="", error="Empty code")
    ]
    failure = classify_failure(
        question="Compute something",
        final_answer="Wrong",
        tool_calls=tool_calls,
        step_count=2
    )
    assert failure == "MALFORMED_TOOL_CALL"

def test_taxonomy_wrong_tool_selection():
    tool_calls = [
        ToolCallRecord(step=1, tool="web_search", args={"query": "server_logs.txt 500 count"}, success=True, output="None")
    ]
    failure = classify_failure(
        question="Read 'benchmarks/data/server_logs.txt' and count 500 status lines.",
        final_answer="0",
        tool_calls=tool_calls,
        step_count=2
    )
    assert failure == "WRONG_TOOL_SELECTION"

def test_taxonomy_premature_termination():
    failure = classify_failure(
        question="Calculate the determinant of matrix",
        final_answer="",
        tool_calls=[],
        step_count=1
    )
    assert failure == "PREMATURE_TERMINATION"

def test_compute_run_metrics():
    traces = [
        EpisodeTrace(
            episode_id="ep_1",
            task_id="t1",
            category="math",
            question="What is 2+2?",
            ground_truth="4",
            final_answer="4",
            is_correct=True,
            status="success",
            step_count=2,
            duration_seconds=1.5,
            tool_calls=[
                ToolCallRecord(step=1, tool="python_repl", args={"code": "print(2+2)"}, success=True, output="4", execution_time=0.1)
            ]
        ),
        EpisodeTrace(
            episode_id="ep_2",
            task_id="t2",
            category="math",
            question="What is 5*5?",
            ground_truth="25",
            final_answer="25",
            is_correct=True,
            status="success",
            step_count=3,
            duration_seconds=2.0,
            tool_calls=[
                ToolCallRecord(step=1, tool="python_repl", args={"code": "prnt(5*5)"}, success=False, output="", error="NameError", execution_time=0.1),
                ToolCallRecord(step=2, tool="python_repl", args={"code": "print(5*5)"}, success=True, output="25", execution_time=0.1)
            ],
            error_recovery_observed=True
        ),
        EpisodeTrace(
            episode_id="ep_3",
            task_id="t3",
            category="search",
            question="Unknown fact",
            ground_truth="Truth",
            final_answer="Falsehood",
            is_correct=False,
            status="failed",
            step_count=2,
            duration_seconds=1.0,
            tool_calls=[
                ToolCallRecord(step=1, tool="web_search", args={"query": "test"}, success=True, output="text", execution_time=0.2)
            ],
            failure_category="HALLUCINATED_TOOL_OUTPUT"
        )
    ]
    summary = compute_run_metrics(traces, mode="test")
    assert summary.total_tasks == 3
    assert summary.completed_tasks == 2
    assert summary.goal_completion_rate == pytest.approx(66.67, 0.01)
    assert summary.step_level_success_rate == pytest.approx(75.0, 0.01) # 3 success out of 4 calls
    assert summary.error_recovery_rate == pytest.approx(100.0, 0.01) # 1 error, and recovered
    assert summary.tool_distribution["python_repl"] == 3
    assert summary.tool_distribution["web_search"] == 1
    assert summary.failure_distribution["HALLUCINATED_TOOL_OUTPUT"] == 1
    assert summary.goal_completion_ci95[0] < summary.goal_completion_rate < summary.goal_completion_ci95[1]

def test_wilson_score_interval():
    from evaluation.metrics import compute_wilson_score_interval
    # 10 out of 20 = 50%
    lower, upper = compute_wilson_score_interval(10, 20)
    assert 29.0 < lower < 31.0
    assert 69.0 < upper < 71.0
    # Boundary: 0 out of 10
    lower_0, upper_0 = compute_wilson_score_interval(0, 10)
    assert lower_0 == 0.0
    assert upper_0 > 0.0
    # Boundary: 10 out of 10
    lower_10, upper_10 = compute_wilson_score_interval(10, 10)
    assert lower_10 < 100.0
    assert upper_10 == 100.0
