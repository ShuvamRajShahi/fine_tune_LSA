"""
04_inference_cuda.py — Inference on CUDA (Windows / Linux)

Drop-in replacement for 04_inference.py (which uses mlx_lm).
The classify_flow() function has the SAME signature and return format,
so downstream code (e.g. your chatbot) needs zero changes.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ──────────────────────── settings ────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = Path("adapters_cuda")
# ──────────────────────────────────────────────────────────────────────

LABELS = ["Benign Status", "DDoS Attack", "Dictionary Attack", "Other Attack"]
SYSTEM = open("data/system_prompt.txt").read()
COLS   = json.load(open("data/feature_cols.json"))


def extract_label(text: str) -> str:
    """Extract label reliably, tolerating markdown bolding and whitespace."""
    clean = text.strip().replace("*", "").replace("`", "")
    for l in LABELS:
        if clean.lower().startswith(l.lower()):
            return l
    first_line = clean.split("\n")[0].lower()
    for l in LABELS:
        if l.lower() in first_line:
            return l
    for l in LABELS:
        if l.lower() in clean.lower():
            return l
    return "Unknown"

# ─────────────────────── load model once ─────────────────────────────
print("Loading model …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=False,
)
if ADAPTER_PATH.exists():
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
model.eval()


@torch.no_grad()
def _generate(prompt: str, max_tokens: int = 30) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_tokens, do_sample=False
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def classify_flow(features: dict) -> dict:
    """
    Classify a single network flow.

    Args:
        features: dict mapping column names to numeric values.

    Returns:
        {"status": "<label>", "raw": "<full model output>"}

    This function has the SAME signature and return format as
    the original 04_inference.py so downstream code is unaffected.
    """
    text = ", ".join(f"{c}={float(features.get(c, 0)):.4g}" for c in COLS)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": "Classify this network flow:\n" + text},
    ]
    prompt = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    out = _generate(prompt)
    status = extract_label(out)
    return {"status": status, "raw": out}


if __name__ == "__main__":
    demo = {c: 0 for c in COLS}
    demo.update({"Rate": 45.2, "IAT": 0.02, "SSH": 1, "TCP": 1})
    print(classify_flow(demo))
    # In your chatbot: if result["status"] != "Benign Status": send_alert(result)

