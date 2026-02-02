from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def python_repl(code: str) -> str:
    """Execute Python code in an isolated subprocess. Always print the result."""
    return "Executed"

llm = ChatOllama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", temperature=0.0)
llm_with_tools = llm.bind_tools([python_repl])

msg = HumanMessage(content="Compute the 15th Fibonacci number using Python.")
response = llm_with_tools.invoke([msg])
print("Content:", response.content)
print("Tool calls:", response.tool_calls)
