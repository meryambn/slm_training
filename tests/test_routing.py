import pytest
import math
from agent.routing import compute_consensus_and_entropy, normalize_candidate, SelfConsistencyRouter

def test_normalize_candidate():
    assert normalize_candidate("The answer is Paris.") == "paris"
    assert normalize_candidate("FINAL ANSWER: $16,135.93") == "16135.93"
    assert normalize_candidate("  Canberra  ") == "canberra"

def test_unanimous_consensus():
    samples = ["Canberra", "canberra", "Canberra."]
    agreement, entropy, modal = compute_consensus_and_entropy(samples)
    assert agreement == 1.0
    assert entropy == 0.0
    assert modal == "canberra"

def test_majority_consensus():
    samples = ["London", "Paris", "London"]
    agreement, entropy, modal = compute_consensus_and_entropy(samples)
    assert agreement == pytest.approx(0.667, 0.01)
    assert entropy > 0.0
    assert modal == "london"

def test_complete_disagreement_entropy():
    samples = ["Alpha", "Beta", "Gamma"]
    agreement, entropy, modal = compute_consensus_and_entropy(samples)
    assert agreement == pytest.approx(0.333, 0.01)
    assert entropy == pytest.approx(round(math.log(3), 3), 0.01)

def test_router_early_exit_on_unanimous():
    router = SelfConsistencyRouter()
    decision = router.evaluate(
        question="What is the capital city of Australia?",
        pre_sampled_candidates=["Canberra", "Canberra", "Canberra"]
    )
    assert decision["action"] == "early_exit"
    assert decision["target_tool"] is None
    assert decision["confidence"] >= 0.90
    assert decision["consensus_agreement"] == 1.0
    assert decision["semantic_entropy"] == 0.0

def test_router_tool_dispatch_on_disagreement():
    router = SelfConsistencyRouter()
    decision = router.evaluate(
        question="What is the capital city of Australia?",
        pre_sampled_candidates=["Canberra", "Sydney", "Melbourne"]
    )
    assert decision["action"] == "route_tool"
    assert decision["consensus_agreement"] < 1.0
    assert decision["semantic_entropy"] > 0.0

def test_router_tool_dispatch_on_tool_flag():
    router = SelfConsistencyRouter()
    decision = router.evaluate(
        question="Compute something difficult",
        pre_sampled_candidates=["UNCERTAIN_NEEDS_TOOL", "UNCERTAIN_NEEDS_TOOL", "123"]
    )
    assert decision["action"] == "route_tool"

def test_router_tool_dispatch_on_file_query():
    router = SelfConsistencyRouter()
    decision = router.evaluate(
        question="Read benchmarks/data/server_logs.txt and count lines.",
        pre_sampled_candidates=["14", "14", "14"]
    )
    assert decision["action"] == "route_tool"
    assert decision["target_tool"] == "file_io"
