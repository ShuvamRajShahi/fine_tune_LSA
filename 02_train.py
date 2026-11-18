"""
02_train.py — QLoRA fine-tuning on CUDA (Windows / Linux + NVIDIA GPU)

Drop-in replacement for 02_train.sh (which uses mlx_lm.lora).
Reads the same data/train.jsonl and data/valid.jsonl produced by the
01_*.py data-preparation scripts.  Saves PEFT adapters to adapters_cuda/.

Hyper-parameters below mirror the original MLX adapter_config.json:
  rank=8, lr=1e-5, 1500 steps, batch_size=1, eval every 150, save every 300.
"""

import json, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ──────────────────────── settings (edit here) ────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"   # base model on HF Hub
DATA_DIR       = Path("data")
ADAPTER_OUT    = Path("adapters_cuda")

LORA_RANK      = 8
LORA_ALPHA     = 20          # scale in MLX config was 20.0
LORA_DROPOUT   = 0.0
NUM_LAYERS     = 16          # target the last 16 transformer layers

LR             = 1e-5
BATCH_SIZE     = 1
GRAD_ACCUM     = 1
MAX_SEQ_LEN    = 1024
MAX_STEPS      = 1500
EVAL_STEPS     = 150
SAVE_STEPS     = 300
SEED           = 0
# ──────────────────────────────────────────────────────────────────────


def load_chat_jsonl(path: Path) -> Dataset:
    """Load a JSONL file of {"messages": [...]} into a HF Dataset."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return Dataset.from_list(rows)


def formatting_func(example):
    """Convert a {"messages": [...]} example to a single prompt string."""
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def build_lora_target_modules(model, num_layers: int):
    """
    Target the attention projections of the last `num_layers` transformer
    layers, matching the original MLX config (--num-layers 16).
    """
    total_layers = model.config.num_hidden_layers
    start = max(0, total_layers - num_layers)
    targets = []
    for i in range(start, total_layers):
        targets.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
        ])
    return targets


# ─────────────────────── 1. load tokenizer ────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────── 2. load model in 4-bit ──────────────────────
print("Loading model in 4-bit NF4 …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=False,
)
model.config.use_cache = False  # required for gradient checkpointing
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ─────────────────────── 3. attach LoRA ──────────────────────────────
target_modules = build_lora_target_modules(model, NUM_LAYERS)
print(f"LoRA on {len(target_modules)} projections across the last {NUM_LAYERS} layers")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────── 4. load data ────────────────────────────────
train_ds = load_chat_jsonl(DATA_DIR / "train.jsonl")
valid_ds = load_chat_jsonl(DATA_DIR / "valid.jsonl")
print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}")

# ─────────────────────── 5. training args ────────────────────────────
training_args = SFTConfig(
    output_dir=str(ADAPTER_OUT),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    fp16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    seed=SEED,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    max_grad_norm=1.0,
    max_length=MAX_SEQ_LEN,
)

# ─────────────────────── 6. train ────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

print("Starting training …")
trainer.train()

# ─────────────────────── 7. save final adapter ───────────────────────
model.save_pretrained(ADAPTER_OUT)
tokenizer.save_pretrained(ADAPTER_OUT)
print(f"\nDone. Adapters saved to {ADAPTER_OUT.resolve()}")

