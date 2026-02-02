"""
Sanity script to verify Ollama connectivity, model availability,
and structured tool calling capability with qwen2.5:7b-instruct.
"""
import time
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct"

def check_ollama_status():
    print(f"[1/3] Checking Ollama server at {OLLAMA_BASE_URL}...", flush=True)
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned status {resp.status_code}: {resp.text}")
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"      Available models in Ollama: {models}", flush=True)
        matching = [m for m in models if MODEL_NAME in m]
        if not matching:
            raise RuntimeError(f"Model '{MODEL_NAME}' not found in Ollama!")
        print(f"      Verified: '{matching[0]}' is present.", flush=True)

def test_inference_and_latency():
    print(f"\n[2/3] Testing inference latency on {MODEL_NAME}...", flush=True)
    start_time = time.perf_counter()
    with httpx.Client(timeout=180.0) as client:
        payload = {
            "model": MODEL_NAME,
            "prompt": "Compute 17 * 19 and explain in one short sentence.",
            "stream": False
        }
        resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        elapsed = time.perf_counter() - start_time
        if resp.status_code != 200:
            raise RuntimeError(f"Inference failed: {resp.text}")
        data = resp.json()
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1e9
        tok_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
        print(f"      Response: {data.get('response', '').strip()}", flush=True)
        print(f"      Wall clock: {elapsed:.2f}s | Tokens: {eval_count} | Speed: {tok_per_sec:.1f} tok/s", flush=True)

def test_tool_calling_format():
    print(f"\n[3/3] Testing tool-calling capability via chat API...", flush=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations in Python",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Python arithmetic expression"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    messages = [
        {"role": "user", "content": "What is the square root of 5476 multiplied by 15?"}
    ]
    with httpx.Client(timeout=180.0) as client:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "stream": False
        }
        resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Chat API failed: {resp.text}")
        msg = resp.json().get("message", {})
        tool_calls = msg.get("tool_calls", [])
        print(f"      Assistant response text: {msg.get('content', '')}", flush=True)
        print(f"      Assistant tool calls: {tool_calls}", flush=True)
        if tool_calls:
            print("      SUCCESS: Model correctly emitted structured tool calls!", flush=True)
        else:
            print("      NOTE: Model responded with plain text without structured tool call.", flush=True)

if __name__ == "__main__":
    check_ollama_status()
    test_inference_and_latency()
    test_tool_calling_format()
