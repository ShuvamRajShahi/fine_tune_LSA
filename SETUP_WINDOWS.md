# Windows + NVIDIA CUDA Setup Guide

> Quick-start guide for running this project on **Windows with an NVIDIA GPU**.
> See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for the full context on what changed and why.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 | Windows 11 |
| GPU | NVIDIA with ≥ 6 GB VRAM | RTX 3060+ / RTX 4060+ |
| NVIDIA Driver | ≥ 528.33 | Latest from nvidia.com |
| CUDA Toolkit | 12.1+ | 12.4 (bundled with PyTorch) |
| Python | 3.10 | 3.11 or 3.12 |

> [!IMPORTANT]
> **Python 3.13** may not yet have full PyTorch CUDA wheel support.
> If `pip install torch` fails, install Python 3.11 or 3.12 and create
> the virtual environment with that version.

---

## Step 1: Create a Clean Virtual Environment

```powershell
cd D:\Study\7\Project\fine_tune_LSA

# Delete the broken old venv (if present)
Remove-Item -Recurse -Force .\.env -ErrorAction SilentlyContinue

# Create a new venv
python -m venv .env

# Activate it
.\.env\Scripts\Activate.ps1
```

---

## Step 2: Install Dependencies

```powershell
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### Verify GPU is detected

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

---

## Step 3: Train (QLoRA Fine-Tuning)

```powershell
python 02_train.py
```

- Loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit quantization
- Trains LoRA adapters using the existing `data/train.jsonl` and `data/valid.jsonl`
- Saves adapters to `adapters_cuda/`
- Takes approximately 15–30 minutes on an RTX 4070

> [!TIP]
> You can adjust hyper-parameters at the top of `02_train.py`.
> The defaults mirror the original MLX training config.

---

## Step 4: Evaluate

```powershell
python 03_eval_cuda.py
```

Prints accuracy and a confusion matrix, same format as the original `03_eval.py`.

---

## Step 5: Run Inference

```powershell
python 04_inference_cuda.py
```

Or import in your own code:

```python
from inference_cuda import classify_flow

result = classify_flow({"Rate": 45.2, "IAT": 0.02, "SSH": 1, "TCP": 1})
print(result)  # {"status": "Dictionary Attack", "raw": "Dictionary Attack. ..."}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `pip` launcher error | Run `python -m pip` instead of `pip` directly |
| `torch.cuda.is_available()` returns `False` | Update NVIDIA drivers; reinstall torch with correct CUDA version |
| `bitsandbytes` import error on Windows | Ensure `bitsandbytes >= 0.43.0` (earlier versions had no Windows support) |
| Out of memory during training | Reduce `max_seq_length` or use `--gradient_checkpointing` (enabled by default) |
| Python 3.13 wheel not found | Use Python 3.11 or 3.12 for the venv |

