"""
Unit tests for Real-World GSM8K Benchmark Loader.
"""
import pytest
from benchmarks.real_benchmarks import (
    extract_gsm8k_ground_truth,
    load_gsm8k_benchmark,
    _fallback_gsm8k_tasks
)

def test_extract_gsm8k_ground_truth():
    # Standard format
    assert extract_gsm8k_ground_truth("She has 10 apples.\n#### 10") == "10"
    
    # Currency and commas
    assert extract_gsm8k_ground_truth("Total cost is $1,250.50.\n#### $1,250.50") == "1250.50"
    assert extract_gsm8k_ground_truth("Revenue is $70,000.\n#### 70,000") == "70000"

    # Negative numbers
    assert extract_gsm8k_ground_truth("Difference is negative.\n#### -42") == "-42"

    # Fallback when #### is omitted
    assert extract_gsm8k_ground_truth("The final value computed is 84") == "84"

def test_load_gsm8k_benchmark_cached_or_fallback():
    tasks = load_gsm8k_benchmark(limit=5)
    assert len(tasks) == 5
    for task in tasks:
        assert task.task_id.startswith("gsm8k_")
        assert len(task.question) > 10
        assert task.ground_truth != ""
        assert task.category == "math_reasoning"
        assert isinstance(task.metadata, dict)

def test_fallback_gsm8k_tasks():
    tasks = _fallback_gsm8k_tasks(limit=3)
    assert len(tasks) == 3
    assert tasks[0].ground_truth == "18"
    assert "eggs" in tasks[0].question.lower()
