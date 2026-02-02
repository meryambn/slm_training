"""
End-to-End QLoRA Fine-Tuning Pipeline for ReAct Mathematical Reasoning SLMs.
Optimized for single 8GB VRAM consumer/workstation GPUs (RTX 2000 Ada, RTX 3070/4060).
"""
import os
# Enable high-speed Rust-based multi-threaded downloads with automatic socket retry
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from training.config import PipelineConfig, QLoRAConfig, TrainingConfig
from training.dataset_loader import prepare_react_math_dataset

def get_device_and_quant_config(qlora_cfg: QLoRAConfig):
    """Detect available hardware acceleration and build appropriate quantization config."""
    cuda_available = torch.cuda.is_available()
    print(f"[Device Detection] CUDA Available: {cuda_available}")
    use_bf16 = False
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        use_bf16 = torch.cuda.is_bf16_supported()
        print(f"[Hardware] Active Device: {gpu_name} ({vram_mb:.0f} MB VRAM, BF16: {use_bf16})")
        
        # Configure 4-bit NF4 quantization for 8GB VRAM with native BF16 compute
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg.use_4bit,
            bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qlora_cfg.use_nested_quant,
        )
        torch_dtype = compute_dtype
        device_map = {"": 0}
    else:
        print("[Hardware Warning] Running on CPU. 4-bit quantization disabled.")
        bnb_config = None
        torch_dtype = torch.float32
        device_map = "cpu"

    return cuda_available, bnb_config, torch_dtype, device_map, use_bf16

def run_training(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    output_dir: str = "training/checkpoints/qwen2.5_7b_react_math",
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    lora_r: int = 64,
    lora_alpha: int = 128,
    dataset_train_size: int = 200,
    dataset_val_size: int = 30,
    dry_run: bool = False
):
    print("==================================================================")
    print("      SLM ReAct Agent QLoRA Fine-Tuning Pipeline (8GB VRAM)       ")
    print("==================================================================")
    print(f"Base Model:       {model_name}")
    print(f"Output Directory: {output_dir}")
    print(f"LoRA Capacity:    r={lora_r}, alpha={lora_alpha}")
    print(f"Epochs / Steps:   {epochs} epochs (max_steps={max_steps})")
    print(f"Effective Batch:  {batch_size * gradient_accumulation_steps}")

    # 1. Ensure training dataset exists
    train_path = Path(train_file) if train_file else Path("training/data/gsm8k_react_train.jsonl")
    val_path = Path(val_file) if val_file else Path("training/data/gsm8k_react_val.jsonl")

    if not train_path.exists() or not val_path.exists():
        print("[Dataset] Local dataset not found. Generating verified GSM8K trajectories...")
        train_path, val_path = prepare_react_math_dataset(
            max_train_samples=dataset_train_size,
            max_val_samples=dataset_val_size
        )

    # 2. Hardware setup
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.qlora.lora_r = lora_r
    pipeline_cfg.qlora.lora_alpha = lora_alpha
    cuda_available, bnb_config, torch_dtype, device_map, use_bf16 = get_device_and_quant_config(pipeline_cfg.qlora)

    if dry_run:
        print("[Dry Run] Dataset verified and hardware configuration checked. Exiting cleanly.")
        return

    # 3. Load Tokenizer
    print(f"[Tokenizer] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Base Model with 4-bit Quantization
    print(f"[Model] Loading base weights ({model_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        token=False
    )

    if cuda_available and bnb_config:
        model = prepare_model_for_kbit_training(model)

    # 5. Configure LoRA Adapter
    lora_cfg = pipeline_cfg.qlora
    peft_config = LoraConfig(
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Load Dataset
    data_files = {"train": str(train_path), "validation": str(val_path)}
    raw_dataset = load_dataset("json", data_files=data_files)
    print(f"[Dataset] Train samples: {len(raw_dataset['train'])}, Val samples: {len(raw_dataset['validation'])}")

    # 7. SFT Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps" if len(raw_dataset["validation"]) > 0 else "no",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataset_text_field="text",
        max_length=pipeline_cfg.training.max_seq_length,
        gradient_checkpointing=pipeline_cfg.training.gradient_checkpointing,
        bf16=use_bf16,
        fp16=(not use_bf16 and cuda_available),
        optim=pipeline_cfg.training.optim if cuda_available else "adamw_torch",
        report_to="none",
        seed=42
    )

    # 8. SFTTrainer Execution
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["validation"],
        processing_class=tokenizer,
    )

    print("\n[Training] Commencing SFT training...")
    start_t = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start_t
    print(f"[Training Complete] Completed in {elapsed:.1f} seconds.")

    # 9. Save final adapter
    final_adapter_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"[Export] Saved trained LoRA adapter to {final_adapter_dir}")
    print(f"[Next Step] Use 'python -m training.export_ollama --adapter-dir {final_adapter_dir}' to export for local Ollama serving.")

def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for ReAct SLMs on Real Math Datasets")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base Hugging Face model ID")
    parser.add_argument("--train-file", type=str, default=None, help="Path to pre-built training JSONL")
    parser.add_argument("--val-file", type=str, default=None, help="Path to pre-built validation JSONL")
    parser.add_argument("--output-dir", type=str, default="training/checkpoints/qwen2.5_7b_react_math", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank dimension (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha scaling factor (default: 128)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--train-size", type=int, default=300, help="Number of GSM8K training examples to synthesize")
    parser.add_argument("--val-size", type=int, default=30, help="Number of GSM8K validation examples to synthesize")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations and dataset without executing forward/backward passes")
    args = parser.parse_args()

    run_training(
        model_name=args.model_id,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dataset_train_size=args.train_size,
        dataset_val_size=args.val_size,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
