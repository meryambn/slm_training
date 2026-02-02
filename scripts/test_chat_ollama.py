from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434", temperature=0.0)
resp = llm.invoke([HumanMessage(content="Say 'Ready' in one word.")])
print(f"ChatOllama response: {resp.content}")
