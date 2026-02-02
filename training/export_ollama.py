"""
Export and Ollama Deployment Pipeline for Fine-Tuned ReAct SLMs.
Merges trained LoRA adapters and generates Ollama Modelfiles for local serving.
"""
import os
import argparse
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = """You are a rigorous mathematical reasoning agent equipped with Python REPL capabilities.
When solving mathematical or algorithmic problems, always write and execute Python code using python_repl.
Always verify calculations with the tool before answering."""

def generate_ollama_modelfile(
    output_dir: Path,
    base_model_ollama: str = "qwen2.5:3b-instruct",
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None
) -> Path:
    """Generate an official Ollama Modelfile for one-command local registration."""
    modelfile_path = output_dir / "Modelfile"

    from_target = merged_model_path if merged_model_path else base_model_ollama
    lines = [
        f"FROM {from_target}",
        'TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>"""',
        f'SYSTEM """{SYSTEM_PROMPT}"""',
        "PARAMETER temperature 0.2",
        "PARAMETER top_p 0.95",
        'PARAMETER stop "<|im_end|>"',
        'PARAMETER stop "<|im_start|>"'
    ]

    if adapter_path and not merged_model_path:
        lines.insert(1, f"ADAPTER {adapter_path}")

    modelfile_path.write_text("\n".join(lines), encoding="utf-8")
    return modelfile_path

def merge_adapter_to_base(
    base_model_name: str,
    adapter_dir: str,
    output_dir: str
):
    """Merge LoRA weights directly into base model weights and save as standard safetensors."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Merge] Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        token=False
    )

    print(f"[Merge] Merging LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    print(f"[Save] Saving full merged model to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=True)
    tokenizer.save_pretrained(out_path)

    modelfile = generate_ollama_modelfile(out_path, merged_model_path=str(out_path.resolve()))
    print(f"[Modelfile] Created Ollama Modelfile at {modelfile}")
    print("\nTo deploy this fine-tuned model directly into your local Ollama runtime, run:")
    print(f"  ollama create qwen2.5-react-math -f {modelfile}")

def main():
    parser = argparse.ArgumentParser(description="Export LoRA Adapter & Build Ollama Modelfile")
    parser.add_argument("--adapter-dir", type=str, default="training/checkpoints/qwen2.5_7b_react_math/final_adapter", help="Directory containing saved LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base HF model ID")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights into standalone model weights")
    parser.add_argument("--output-dir", type=str, default="training/checkpoints/qwen2.5_7b_react_math_merged", help="Output directory for merged weights")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_adapter_to_base(args.base_model, str(adapter_path), str(output_path))
    else:
        modelfile = generate_ollama_modelfile(output_path, base_model_ollama="qwen2.5:7b-instruct", adapter_path=str(adapter_path.resolve()))
        print(f"[Export] Created Ollama Modelfile at {modelfile}")
        print("\nTo register with local Ollama:")
        print(f"  ollama create qwen2.5-react-math -f {modelfile}")

if __name__ == "__main__":
    main()
