# MLX → CUDA Migration Plan

> **Goal**: Run this fine-tuning project on **Windows + NVIDIA GPU (RTX 4070, 8 GB VRAM)**
> instead of macOS + Apple Silicon (MLX).
>
> **Principle**: Change as little as possible — keep the original files intact, keep the
> data pipeline and data format **100% unchanged**, and produce drop-in replacements
> only for the files that import `mlx`.

---

## What Changed and Why

| Original File | MLX Dependent? | Action Taken |
|---|---|---|
| `01_build_from_folders.py` | ❌ No | **Untouched** — pure pandas/numpy |
| `01_label_and_build.py` | ❌ No | **Untouched** — pure pandas/numpy |
| `data/*.jsonl` | ❌ No | **Untouched** — standard chat-format JSONL |
| `02_train.sh` | ✅ Yes (`mlx_lm.lora`) | **Kept as-is** (for macOS). Added `02_train.py` for Windows/CUDA |
| `03_eval.py` | ✅ Yes (`mlx_lm`) | **Kept as-is** (for macOS). Added `03_eval_cuda.py` for Windows |
| `04_inference.py` | ✅ Yes (`mlx_lm`) | **Kept as-is** (for macOS). Added `04_inference_cuda.py` for Windows |
| `requirements.txt` | ✅ Yes (`mlx*`) | **Renamed → `requirements-mac.txt`**. New `requirements.txt` for Windows/CUDA |
| `adapters/` | ⚠️ MLX format | Need to **retrain** on CUDA (adapters saved in PEFT-compatible format) |

---

## Migration Steps

### Step 1 — Fix Python Environment ✅

```powershell
# From inside fine_tune_LSA/
deactivate                              # if old venv is active
Remove-Item -Recurse -Force .\.env      # delete broken venv
python -m venv .env                     # create fresh venv
.\.env\Scripts\Activate.ps1             # activate
```

### Step 2 — Install CUDA Dependencies ✅

```powershell
# PyTorch with CUDA 12.4 (matches your RTX 4070 driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then the rest
pip install -r requirements.txt
```

> [!NOTE]
> If torch installation fails on Python 3.13, create the venv with Python 3.11 or 3.12
> instead — PyTorch's CUDA wheels have the best compatibility there.

### Step 3 — Train with QLoRA on CUDA ✅

```powershell
python 02_train.py
```

This script:
- Loads `Qwen/Qwen2.5-7B-Instruct` in **4-bit NF4 quantization** (fits in 8 GB VRAM)
- Applies **LoRA** with the same hyper-parameters as the original MLX config
  (rank=8, 16 target layers, lr=1e-5, 1500 steps, batch_size=1)
- Reads `data/train.jsonl` and `data/valid.jsonl` (unchanged format)
- Saves PEFT adapters to `adapters_cuda/`

### Step 4 — Evaluate ✅

```powershell
python 03_eval_cuda.py
```

Same logic as `03_eval.py` but uses `transformers` + `peft` instead of `mlx_lm`.

### Step 5 — Run Inference ✅

```powershell
python 04_inference_cuda.py
```

Same `classify_flow()` function signature and output format as `04_inference.py`.

---

## Model Mapping

| MLX (macOS) | Hugging Face (CUDA) | Notes |
|---|---|---|
| `mlx-community/Qwen3-8B-4bit` | `Qwen/Qwen2.5-7B-Instruct` | Same architecture family. The MLX version is a community-quantized mirror. On CUDA we quantize on-the-fly with `bitsandbytes` NF4. |

---

## Directory Layout After Migration

```
fine_tune_LSA/
├── MIGRATION_PLAN.md          ← this file
├── SETUP_WINDOWS.md           ← step-by-step Windows setup guide
│
├── 01_build_from_folders.py   ← UNCHANGED (data prep)
├── 01_label_and_build.py      ← UNCHANGED (data prep)
│
├── 02_train.sh                ← UNCHANGED (macOS MLX training)
├── 02_train.py                ← NEW: Windows/CUDA QLoRA training
│
├── 03_eval.py                 ← UNCHANGED (macOS MLX eval)
├── 03_eval_cuda.py            ← NEW: Windows/CUDA evaluation
│
├── 04_inference.py            ← UNCHANGED (macOS MLX inference)
├── 04_inference_cuda.py       ← NEW: Windows/CUDA inference
│
├── requirements-mac.txt       ← RENAMED from original requirements.txt
├── requirements.txt           ← NEW: Windows/CUDA dependencies
│
├── data/                      ← UNCHANGED
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   ├── system_prompt.txt
│   └── feature_cols.json
│
├── adapters/                  ← UNCHANGED (MLX adapters from macOS)
└── adapters_cuda/             ← NEW: PEFT adapters from CUDA training
```

---

## Key Design Decisions

1. **Separate adapter directories** (`adapters/` vs `adapters_cuda/`):
   MLX and PEFT use different weight-key naming conventions. Keeping them separate
   avoids confusion and lets both platforms coexist.

2. **Separate script files** (not if/else blocks):
   Cleaner than littering every file with `if platform == "darwin"` branches.
   Each script is self-contained and easy to understand.

3. **Same data format**:
   Both platforms read the exact same `data/*.jsonl` files. No conversion needed.

4. **Same hyper-parameters**:
   The CUDA training script mirrors the MLX config: LoRA rank 8, lr 1e-5, 1500 steps,
   batch size 1, gradient checkpointing, eval every 150 steps, save every 300 steps.

