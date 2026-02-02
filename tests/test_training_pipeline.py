"""
Unit tests for ReAct Trajectory Dataset Synthesis, QLoRA Configs, and Ollama Export.
"""
from pathlib import Path
import pytest
from tools.python_repl import PythonREPLTool
from training.config import PipelineConfig, QLoRAConfig, TrainingConfig
from training.dataset_loader import (
    extract_expressions_from_gsm8k,
    synthesize_python_code,
    build_react_trajectory
)
from training.export_ollama import generate_ollama_modelfile

def test_extract_expressions_from_gsm8k():
    solution = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs. Then 9 * 2 = <<9*2=18>>18 dollars.\n#### 18"
    exprs = extract_expressions_from_gsm8k(solution)
    assert len(exprs) == 2
    assert exprs[0] == ("16-3-4", "9")
    assert exprs[1] == ("9*2", "18")

def test_synthesize_python_code():
    exprs = [("16 - 3 - 4", "9"), ("9 * 2", "18")]
    code = synthesize_python_code("How many dollars?", exprs, "18")
    assert "step_1 = 16 - 3 - 4" in code
    assert "step_2 = 9 * 2" in code
    assert "print(step_2)" in code

def test_build_react_trajectory():
    repl = PythonREPLTool(timeout=5)
    q = "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes. How much did she earn?"
    sol = "She earns 12 / 60 = <<12/60=0.2>>0.2 per min. Then 0.2 * 50 = <<0.2*50=10>>10 dollars.\n#### 10"
    gt = "10"
    
    trajectory = build_react_trajectory(q, sol, gt, repl)
    assert trajectory is not None
    assert trajectory["ground_truth"] == "10"
    assert "python_repl" in trajectory["messages"][2]["content"]
    assert "Action: python_repl" in trajectory["messages"][2]["content"]
    assert "Final Answer: 10" in trajectory["messages"][2]["content"]

def test_qlora_and_training_config_defaults():
    cfg = PipelineConfig()
    assert cfg.qlora.use_4bit is True
    assert "7B" in cfg.qlora.model_name_or_path
    assert cfg.qlora.lora_r == 64
    assert cfg.qlora.lora_alpha == 128
    assert "q_proj" in cfg.qlora.lora_target_modules
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.gradient_checkpointing is True

def test_generate_ollama_modelfile(tmp_path: Path):
    modelfile = generate_ollama_modelfile(
        output_dir=tmp_path,
        base_model_ollama="qwen2.5:3b-instruct",
        adapter_path="./checkpoints/final_adapter"
    )
    assert modelfile.exists()
    content = modelfile.read_text(encoding="utf-8")
    assert "FROM qwen2.5:3b-instruct" in content
    assert "ADAPTER ./checkpoints/final_adapter" in content
    assert "SYSTEM" in content
    assert "PARAMETER temperature" in content
