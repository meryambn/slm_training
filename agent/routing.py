"""
Uncertainty-Aware Tool Router based on Semantic Self-Consistency and Entropy.
Samples k candidate thoughts/answers at temperature > 0 to quantify epistemic
uncertainty via consensus agreement and semantic entropy.
"""
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import math
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def normalize_candidate(s: str) -> str:
    """Normalize candidate answer string for semantic clustering."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"^(the answer is|final answer:?|result:?)\s*", "", s)
    s = s.replace("$", "").replace(",", "").replace("%", "")
    s = s.rstrip(".").strip()
    return s

def compute_consensus_and_entropy(samples: List[str]) -> Tuple[float, float, str]:
    """
    Computes consensus agreement ratio and semantic entropy across k candidate samples.
    Returns:
        (consensus_agreement, semantic_entropy, modal_answer)
    """
    if not samples:
        return 0.0, 0.0, ""

    normalized = [normalize_candidate(s) for s in samples]
    k = len(normalized)
    counts = Counter(normalized)
    modal_ans, max_count = counts.most_common(1)[0]
    consensus_agreement = max_count / k

    # Compute Shannon entropy H = -sum(p * ln(p))
    entropy = 0.0
    for count in counts.values():
        p = count / k
        if p > 0:
            entropy -= p * math.log(p)

    return round(consensus_agreement, 3), round(entropy, 3), modal_ans

class SelfConsistencyRouter:
    """
    Evaluates epistemic uncertainty via stochastic self-consistency sampling.
    If consensus agreement is unanimous (A = 1.0, Entropy = 0.0), task is solved directly.
    If disagreement or uncertainty is detected (A < 1.0, Entropy > 0), routes to ReAct tools.
    """
    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        k_samples: int = 3,
        temperature: float = 0.7,
        confidence_threshold: float = 0.85
    ):
        self.k_samples = k_samples
        self.confidence_threshold = confidence_threshold
        self.llm_sampling = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )

    def sample_candidates(self, question: str) -> List[str]:
        """Generate k independent candidate answers at non-zero temperature."""
        prompt = (
            f"Provide a direct, concise one-line factual answer to the following question. "
            f"If the question involves reading local files (e.g., .csv, .txt, .json) or computing exact math that requires a script, "
            f"reply strictly with 'UNCERTAIN_NEEDS_TOOL'.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        candidates = []
        for _ in range(self.k_samples):
            try:
                resp = self.llm_sampling.invoke([HumanMessage(content=prompt)])
                ans = resp.content.strip().split("\n")[0].strip()
                candidates.append(ans)
            except Exception:
                candidates.append("UNCERTAIN_NEEDS_TOOL")
        return candidates

    def evaluate(self, question: str, pre_sampled_candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates uncertainty on question. Accepts optional pre_sampled_candidates
        for fast deterministic testing.
        """
        candidates = pre_sampled_candidates if pre_sampled_candidates is not None else self.sample_candidates(question)
        agreement, entropy, modal_ans = compute_consensus_and_entropy(candidates)

        # Check if any candidate flagged tool requirement
        needs_tool_flag = any("uncertain_needs_tool" in c.lower() for c in candidates)
        
        # Check for obvious external data indicators
        q_lower = question.lower()
        has_file_ref = any(ext in q_lower for ext in [".txt", ".csv", ".json", "benchmarks/data"])
        has_math_ref = any(m in q_lower for m in ["mod ", "determinant", "prime", "correlation", "standard deviation", "compound interest"])

        # Early exit allowed ONLY if unanimous agreement, zero entropy, and no tool flag
        if agreement >= 1.0 and entropy == 0.0 and not needs_tool_flag and not has_file_ref and not has_math_ref:
            return {
                "action": "early_exit",
                "target_tool": None,
                "confidence": 0.95,
                "consensus_agreement": agreement,
                "semantic_entropy": entropy,
                "sample_answers": candidates,
                "modal_answer": modal_ans,
                "reasoning": f"Unanimous self-consistency consensus (A={agreement}, H={entropy}) without tool dependency."
            }
        else:
            target = "file_io" if has_file_ref else ("python_repl" if has_math_ref else "react_tools")
            return {
                "action": "route_tool",
                "target_tool": target,
                "confidence": round(agreement * (1.0 - min(entropy, 1.0)), 3),
                "consensus_agreement": agreement,
                "semantic_entropy": entropy,
                "sample_answers": candidates,
                "modal_answer": modal_ans,
                "reasoning": f"Uncertainty detected (agreement={agreement}, entropy={entropy}, needs_tools={needs_tool_flag or has_file_ref or has_math_ref}). Routing to ReAct tools."
            }
