"""
Single task end-to-end verification script for ReAct agent.
"""
from agent.react_graph import ReActAgent
import json

def run_single_math():
    print("=== Testing ReAct Agent on Multi-Step Math Task ===")
    agent = ReActAgent()
    question = "What is the sum of all prime numbers between 100 and 150?"
    print(f"Task: {question}")
    
    result = agent.run(question, task_id="test_prime_sum")
    
    print(f"\nFinal Status: {result.get('status')}")
    print(f"Step Count: {result.get('step_count')}")
    print(f"Final Answer: {result.get('final_answer')}")
    print(f"Execution Time: {result.get('execution_total_time', 0):.2f}s")
    print("\nTool Call History:")
    for step in result.get("tool_call_history", []):
        print(f"  Step {step['step']}: Tool={step['tool']}, Args={step['args']}")
        print(f"    Output: {step['result']['output'][:100]}")
        print(f"    Success: {step['result']['success']}, Time: {step['result']['execution_time']}s")

if __name__ == "__main__":
    run_single_math()
