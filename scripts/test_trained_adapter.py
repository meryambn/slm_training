"""
Test script to run the fine-tuned ReAct SLM LoRA adapter locally.
Runs safely on an 8GB laptop GPU (uses ~4.5GB VRAM in 4-bit).
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_DIR = Path("qwen2.5_7b_react_math_adapter")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def test_inference(question: str):
    print("==========================================================")
    print("   Testing Fine-Tuned ReAct SLM (Qwen2.5-7B + LoRA)      ")
    print("==========================================================")
    print(f"Loading Base Model: {BASE_MODEL} in 4-bit...")

    # 4-bit quantization keeps total VRAM under ~4.5GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Attaching LoRA Adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    system_prompt = (
        "You are a mathematical reasoning assistant with tool-using capabilities. "
        "To solve the problem accurately, write and execute Python code using python_repl. "
        "Always wrap your calculation logic in python_repl and print the final numeric answer."
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print(f"\nUser Question:\n{question}\n")
    print("Generating ReAct Reasoning Trajectory...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\nModel Response:")
    print(generated_text)
    print("==========================================================")

if __name__ == "__main__":
    sample_question = (
    "Calculate the compound interest on an initial investment of $250,000 "
    "at an annual rate of 7.25% compounded monthly for 18 years, "
    "and find the total interest earned rounded to the nearest cent."
)

    test_inference(sample_question)
