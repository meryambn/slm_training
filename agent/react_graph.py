"""
LangGraph ReAct Agent Implementation.
Executes an iterative Think -> Act -> Observe loop with Ollama Qwen2.5.
Supports strictly typed AgentState, Zero-Shot Direct arm, Baseline ReAct,
and Uncertainty-Aware Self-Consistency Routing.
"""
import time
from typing import Dict, Any, List, Optional, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from tools.python_repl import PythonREPLTool
from tools.file_io import FileIOTool
from tools.web_search import WebSearchTool
from agent.state import AgentState
from agent.routing import SelfConsistencyRouter

# Initialize core tool instances
_repl_instance = PythonREPLTool(timeout=10)
_file_io_instance = FileIOTool()
_web_search_instance = WebSearchTool(default_max_results=4)

@tool
def python_repl(code: str) -> str:
    """Execute Python code in an isolated subprocess. Always use print() to output results."""
    res = _repl_instance.execute(code)
    if res.success:
        return res.output
    return f"Execution Error: {res.error}"

@tool
def file_io(action: str, path: str, max_lines: int = 200) -> str:
    """Inspect and read files in the local workspace/benchmark data directory. Actions: read_file, list_dir, read_csv_header."""
    res = _file_io_instance.execute(action=action, path=path, max_lines=max_lines)
    if res.success:
        return res.output
    return f"FileIO Error: {res.error}"

@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the live web for factual information, current data, or entity facts."""
    res = _web_search_instance.execute(query=query, max_results=max_results)
    if res.success:
        return res.output
    return f"WebSearch Error: {res.error}"

TOOLS_MAP = {
    "python_repl": _repl_instance,
    "file_io": _file_io_instance,
    "web_search": _web_search_instance
}

ALL_TOOLS = [python_repl, file_io, web_search]

SYSTEM_PROMPT = """You are a rigorous, highly capable ReAct research assistant.
Solve the given problem step-by-step.
You have access to the following tools:
- python_repl(code: str): Run Python code to perform arithmetic, statistics, algorithmic computation, or parsing. Always use print() to show answers.
- file_io(action: str, path: str, max_lines: int): Read files, list directory contents, or inspect CSV headers in the benchmark data directory.
- web_search(query: str): Search the web for specific factual knowledge.

Guidelines:
1. Reason carefully about what information you need.
2. If computation is needed, ALWAYS use python_repl rather than doing mental math.
3. If inspecting files, use file_io with appropriate paths like 'benchmarks/data/...'.
4. When you have found the definitive answer, provide it clearly at the very end as:
FINAL ANSWER: <concise answer>
"""

class ReActAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_steps: int = 8,
        k_samples: int = 3
    ):
        self.max_steps = max_steps
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.router = SelfConsistencyRouter(
            model_name=model_name,
            base_url=base_url,
            k_samples=k_samples,
            temperature=0.7
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        # Strict typed state using AgentState schema
        builder = StateGraph(AgentState)

        builder.add_node("reason", self._reason_node)
        builder.add_node("execute_tools", self._execute_tools_node)

        builder.set_entry_point("reason")

        builder.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "execute_tools": "execute_tools",
                "end": END
            }
        )
        builder.add_edge("execute_tools", "reason")

        return builder.compile()

    def _reason_node(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        step_count = state.get("step_count", 0) + 1
        
        ai_response = self.llm_with_tools.invoke(messages)

        # Check for FINAL ANSWER in content
        content = ai_response.content or ""
        final_ans = None
        if "FINAL ANSWER:" in content:
            final_ans = content.split("FINAL ANSWER:")[-1].strip().split("\n")[0].strip()
        elif not getattr(ai_response, "tool_calls", None) and step_count > 1:
            lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
            if lines:
                final_ans = lines[-1]

        return {
            "messages": [ai_response],
            "step_count": step_count,
            "final_answer": final_ans
        }

    def _execute_tools_node(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        new_history = []
        new_tool_messages = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            call_id = call.get("id", f"call_{len(state.get('tool_call_history', [])) + len(new_history)}")
            
            start_t = time.perf_counter()
            if tool_name in TOOLS_MAP:
                raw_res = TOOLS_MAP[tool_name].execute(**tool_args)
                obs = raw_res.output if raw_res.success else f"Error: {raw_res.error}"
                res_dict = raw_res.to_dict()
            else:
                obs = f"Error: Tool '{tool_name}' not recognized."
                res_dict = {"success": False, "output": "", "error": obs, "execution_time": 0.0}
            
            elapsed = time.perf_counter() - start_t
            
            new_history.append({
                "step": state.get("step_count", 0),
                "tool": tool_name,
                "args": tool_args,
                "result": res_dict,
                "timestamp": time.time()
            })

            new_tool_messages.append(
                ToolMessage(content=obs, tool_call_id=call_id)
            )

        return {
            "messages": new_tool_messages,
            "tool_call_history": new_history
        }

    def _should_continue(self, state: AgentState) -> Literal["execute_tools", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        step_count = state.get("step_count", 0)
        
        # If final answer already produced or max steps reached
        if state.get("final_answer"):
            return "end"
        if step_count >= self.max_steps:
            return "end"
        
        # If model requested tool calls
        if getattr(last_message, "tool_calls", None):
            return "execute_tools"

        return "end"

    def run(
        self,
        question: str,
        task_id: str = "task_0",
        mode: str = "baseline",
        pre_sampled_candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute agent with specified evaluation arm:
          - 'zero_shot': Direct prompt, no tools allowed (lower baseline)
          - 'baseline': Standard ReAct graph loop without early exit
          - 'routed': Self-consistency uncertainty evaluation before deciding whether to early-exit or invoke ReAct
        """
        start_time = time.perf_counter()

        # Arm 1: Zero-Shot Direct (Lower Baseline reference)
        if mode == "zero_shot":
            prompt = (
                f"You are a helpful reasoning assistant. Answer the following question directly and concisely.\n"
                f"Question: {question}\n"
                f"FINAL ANSWER: <answer>"
            )
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            elapsed = time.perf_counter() - start_time
            content = resp.content or ""
            final_ans = content.split("FINAL ANSWER:")[-1].strip().split("\n")[0].strip() if "FINAL ANSWER:" in content else content.strip()
            return {
                "task_id": task_id,
                "question": question,
                "messages": [HumanMessage(content=prompt), resp],
                "step_count": 1,
                "max_steps": 1,
                "tool_call_history": [],
                "final_answer": final_ans,
                "status": "success" if final_ans else "failed",
                "confidence_score": 1.0,
                "consensus_agreement": 1.0,
                "semantic_entropy": 0.0,
                "sample_answers": [],
                "router_decision": {"action": "zero_shot_direct"},
                "execution_start_time": start_time,
                "execution_total_time": elapsed
            }

        # Arm 2: Uncertainty-Routed ReAct
        router_decision = None
        agreement = 0.0
        entropy = 0.0
        candidates = []
        if mode == "routed":
            router_decision = self.router.evaluate(question, pre_sampled_candidates=pre_sampled_candidates)
            agreement = router_decision.get("consensus_agreement", 0.0)
            entropy = router_decision.get("semantic_entropy", 0.0)
            candidates = router_decision.get("sample_answers", [])

            if router_decision["action"] == "early_exit":
                elapsed = time.perf_counter() - start_time
                return {
                    "task_id": task_id,
                    "question": question,
                    "messages": [HumanMessage(content=question), AIMessage(content=f"FINAL ANSWER: {router_decision['modal_answer']}")],
                    "step_count": 1,
                    "max_steps": self.max_steps,
                    "tool_call_history": [],
                    "final_answer": router_decision["modal_answer"],
                    "status": "success",
                    "confidence_score": router_decision["confidence"],
                    "consensus_agreement": agreement,
                    "semantic_entropy": entropy,
                    "sample_answers": candidates,
                    "router_decision": router_decision,
                    "execution_start_time": start_time,
                    "execution_total_time": elapsed
                }

        # Arm 3: Standard ReAct Graph Execution
        initial_state: AgentState = {
            "task_id": task_id,
            "question": question,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question)
            ],
            "step_count": 0,
            "max_steps": self.max_steps,
            "tool_call_history": [],
            "final_answer": None,
            "status": "running",
            "confidence_score": 1.0,
            "consensus_agreement": agreement,
            "semantic_entropy": entropy,
            "sample_answers": candidates,
            "router_decision": router_decision,
            "execution_start_time": start_time,
            "execution_total_time": 0.0
        }

        final_state = self.graph.invoke(initial_state)
        elapsed = time.perf_counter() - start_time

        # Extract final answer if not cleanly parsed
        if not final_state.get("final_answer"):
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    c = msg.content
                    if "FINAL ANSWER:" in c:
                        final_state["final_answer"] = c.split("FINAL ANSWER:")[-1].strip().split("\n")[0].strip()
                        break
                    else:
                        lines = [line.strip() for line in c.strip().split("\n") if line.strip()]
                        if lines:
                            final_state["final_answer"] = lines[-1]
                            break

        status = "success" if final_state.get("final_answer") else "failed"
        final_state["status"] = status
        final_state["execution_total_time"] = elapsed
        final_state["router_decision"] = router_decision
        final_state["consensus_agreement"] = agreement
        final_state["semantic_entropy"] = entropy
        final_state["sample_answers"] = candidates
        return final_state
