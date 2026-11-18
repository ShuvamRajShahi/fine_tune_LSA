"""
03_eval_cuda.py — Evaluate fine-tuned model on CUDA (Windows / Linux)

Drop-in replacement for 03_eval.py (which uses mlx_lm).
Same output format: accuracy + confusion matrix printed to stdout.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ──────────────────────── settings ────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = Path("adapters_cuda")
TEST_FILE    = Path("data/test.jsonl")
MAX_TOKENS   = 30
# ──────────────────────────────────────────────────────────────────────

LABELS = ["Benign Status", "DDoS Attack", "Dictionary Attack", "Other Attack"]


def head(text):
    """Return the first matching label, or 'Unknown'."""
    return next((l for l in LABELS if text.startswith(l)), "Unknown")


# ─────────────────────── load model ──────────────────────────────────
print("Loading model …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=False,
)
if ADAPTER_PATH.exists():
    print(f"Loading adapters from {ADAPTER_PATH} …")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
else:
    print(f"WARNING: {ADAPTER_PATH} not found — running base model only")

model.eval()

# ─────────────────────── generate helper ─────────────────────────────
@torch.no_grad()
def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=MAX_TOKENS, do_sample=False
    )
    # decode only the new tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────── evaluate ────────────────────────────────────
rows = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8")]
rows = rows[::max(1, len(rows) // 300)][:300]  # spread sample across the test set

conf = Counter()
for i, r in enumerate(rows):
    m = r["messages"]
    prompt = tokenizer.apply_chat_template(
        m[:2], add_generation_prompt=True, tokenize=False
    )
    pred = head(generate(prompt))
    gold = head(m[2]["content"])
    conf[(gold, pred)] += 1
    if (i + 1) % 50 == 0:
        print(f"  evaluated {i + 1}/{len(rows)} …")

# ─────────────────────── print results ───────────────────────────────
correct = sum(v for (g, p), v in conf.items() if g == p)
print(f"\naccuracy: {round(correct / len(rows), 3)}")
for (g, p), v in sorted(conf.items()):
    print(f"true={g:18s} pred={p:18s} {v}")

