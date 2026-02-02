"""
Configuration schemas for QLoRA fine-tuning and dataset preparation.
Optimized for consumer and workstation GPUs with 8GB VRAM (e.g. RTX 2000 Ada).
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QLoRAConfig:
    # Model configuration (Scaled to 7B parameters)
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "bfloat16"  # "bfloat16" native to Ada Lovelace / Qwen
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # LoRA hyper-parameters (Scaled capacity: r=64, alpha=128 ~62M trainable parameters)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainingConfig:
    # Data paths
    train_file: str = "training/data/gsm8k_react_train.jsonl"
    val_file: str = "training/data/gsm8k_react_val.jsonl"
    output_dir: str = "training/checkpoints/qwen2.5_7b_react_math"

    # Optimization parameters tailored for 8GB VRAM
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Sequence length & Memory optimizations
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    
    # Telemetry and checkpoint saving
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42

@dataclass
class PipelineConfig:
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
