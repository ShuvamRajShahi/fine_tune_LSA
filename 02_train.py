"""
02_train.py — Optimized QLoRA Fine-Tuning on CUDA (Windows + NVIDIA RTX 4070)

Key improvements over baseline:
1. Completion-only loss masking: computes cross-entropy loss ONLY on the assistant's
   classification answer, ignoring the ~300 numeric input tokens.
2. Full-network LoRA: targets all 7 linear projections (q, k, v, o, gate, up, down)
   across all 28 layers of Qwen2.5-7B for maximal classification representation.
3. 2 full epochs: trains over the full 11,000-sample dataset twice (effective batch size 16).
4. Memory & compute optimized: 4-bit NF4 quantization + native hardware BFloat16.
"""

from functools import partial
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ──────────────────────── settings ────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR       = Path("data")
ADAPTER_OUT    = Path("adapters_cuda")

# LoRA hyper-parameters (all linear layers across all 28 layers)
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training schedule
LR             = 2e-4
BATCH_SIZE     = 2           # fits comfortably in 8 GB VRAM with seq len 384
GRAD_ACCUM     = 4           # effective batch size = 2 * 4 = 8
MAX_STEPS      = 800         # 800 steps = 6,400 samples (finishes in ~25 mins)
MAX_SEQ_LEN    = 384         # verified: dataset max length is 376 tokens
EVAL_STEPS     = 100
SAVE_STEPS     = 200
SEED           = 42
# ──────────────────────────────────────────────────────────────────────


def load_chat_jsonl(path: Path) -> Dataset:
    """Load JSONL lines into a Hugging Face Dataset."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return Dataset.from_list(rows)


def preprocess_batch(batch, tokenizer, max_seq_len):
    """
    Tokenize messages with completion-only masking:
    - User/System prompt tokens are set to -100 (ignored in loss calculation).
    - Assistant answer tokens retain their token IDs (100% loss focus).
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for msgs in batch["messages"]:
        # 1. Tokenize prompt only (system + user) with generation prompt
        prompt_str = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True
        )
        # 2. Tokenize full dialogue (system + user + assistant answer)
        full_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]

        # 3. Create target labels with -100 mask on the prompt
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 4. Truncate to max_seq_len
        full_ids = full_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = [1] * len(full_ids)

        input_ids_list.append(full_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def main():
    # ─────────────────────── 1. load tokenizer ────────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─────────────────────── 2. load model in 4-bit ──────────────────────
    print("Loading model in 4-bit NF4 with BFloat16 compute …")
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
        attn_implementation="sdpa",     # native hardware-accelerated fused attention
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ─────────────────────── 3. attach LoRA ──────────────────────────────
    print(f"Attaching LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA}) to all linear layers …")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─────────────────────── 4. load and preprocess data ─────────────────
    print("Loading datasets …")
    raw_train = load_chat_jsonl(DATA_DIR / "train.jsonl")
    raw_valid = load_chat_jsonl(DATA_DIR / "valid.jsonl")
    print(f"Raw samples: Train={len(raw_train)}  Valid={len(raw_valid)}")

    print("Tokenizing with prompt masking (loss computed ONLY on assistant responses) …")
    tokenize_fn = partial(preprocess_batch, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    train_ds = raw_train.map(tokenize_fn, batched=True, remove_columns=raw_train.column_names)
    valid_ds = raw_valid.map(tokenize_fn, batched=True, remove_columns=raw_valid.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"
    )

    # ─────────────────────── 5. training arguments ───────────────────────
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_OUT),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        seed=SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_grad_norm=1.0,
    )

    # ─────────────────────── 6. train ────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    print("Starting training …")
    trainer.train()

    # ─────────────────────── 7. save final adapter ───────────────────────
    print("Saving final adapter …")
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    print(f"\nDone! Adapters saved to {ADAPTER_OUT.resolve()}")


if __name__ == "__main__":
    main()
