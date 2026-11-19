import json, sys, random
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- CELL 1: settings you can tune ----------
CSV_FILES = sys.argv[1:] or ["DictionaryBruteForce.csv"] #or ["access_log.csv"]  # pass several CSVs: benign, ddos, dictionary
OUT_DIR = Path("data")
MAX_PER_CLASS = 3000          # keeps classes balanced
SEED = 42

T = dict(
    ddos_rate=1000.0,         # packets/sec above this = flood-like
    ddos_iat=0.001,           # inter-arrival time (sec) below this = flood-like
    dict_rate_min=5.0,        # brute force: steady, not necessarily huge
    dict_rate_max=1000.0,     # above this we treat it as DDoS instead
    dict_iat_max=0.5,         # repeated attempts arrive quickly
)

SYSTEM = ("You are a network security analyst. Given numeric network-flow features, "
          "reply with exactly one status: 'Benign Status', 'DDoS Attack' or 'Dictionary Attack', "
          "followed by a one-line reason.")

# ---------- CELL 2: helpers ----------
def col(df, name):
    """Find a column ignoring case/spaces; return zeros if absent."""
    norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    key = name.lower().replace(" ", "_")
    return df[norm[key]].astype(float) if key in norm else pd.Series(0.0, index=df.index)

def label_rows(df):
    rate, iat = col(df, "Rate"), col(df, "IAT")
    ssh, telnet = col(df, "SSH"), col(df, "Telnet")
    syn, udp, icmp = col(df, "syn_flag_number"), col(df, "UDP"), col(df, "ICMP")
    syn_cnt = col(df, "syn_count")

    flood_sig = (syn > 0) | (udp > 0) | (icmp > 0) | (syn_cnt > 5)
    is_ddos = (rate >= T["ddos_rate"]) & (iat <= T["ddos_iat"]) & flood_sig
    is_dict = (((ssh > 0) | (telnet > 0)) &
               (rate >= T["dict_rate_min"]) & (rate < T["dict_rate_max"]) &
               (iat <= T["dict_iat_max"]) & ~is_ddos)

    labels = np.where(is_ddos, "DDoS Attack",
             np.where(is_dict, "Dictionary Attack", "Benign Status"))
    reasons = np.where(
        is_ddos, "Very high packet rate with near-zero inter-arrival time and flood-type flags/protocol.",
        np.where(is_dict, "Rapid repeated SSH/Telnet connection attempts at a steady rate (login guessing).",
                 "Rate and timing look normal; no flood or repeated-login pattern."))
    return labels, reasons

def row_to_text(row, cols):
    return ", ".join(f"{c}={round(float(row[c]), 4)}" for c in cols)

# ---------- CELL 3: build the dataset ----------
OUT_DIR.mkdir(exist_ok=True)
examples = []
for f in CSV_FILES:
    df = pd.read_csv(f)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n{f}: {len(df)} rows, {len(num_cols)} numeric cols")
    print(df[[c for c in df.columns if c.strip().lower() in ("rate", "iat")]].describe(
        percentiles=[.5, .9, .99]).round(4))          # use this to tune thresholds
    labels, reasons = label_rows(df)
    for i in range(len(df)):
        examples.append({
            "label": labels[i],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Classify this network flow:\n" + row_to_text(df.iloc[i], num_cols)},
                {"role": "assistant", "content": f"{labels[i]}. Reason: {reasons[i]}"},
            ]})

# balance classes
random.seed(SEED)
by_class = {}
for e in examples: by_class.setdefault(e["label"], []).append(e)
print("\nClass counts before balancing:", {k: len(v) for k, v in by_class.items()})
balanced = []
for k, v in by_class.items():
    random.shuffle(v); balanced += v[:MAX_PER_CLASS]
random.shuffle(balanced)

# split 80/10/10 and write (mlx-lm wants train.jsonl / valid.jsonl / test.jsonl)
n = len(balanced); a, b = int(n * .8), int(n * .9)
for name, part in [("train", balanced[:a]), ("valid", balanced[a:b]), ("test", balanced[b:])]:
    with open(OUT_DIR / f"{name}.jsonl", "w") as fh:
        for e in part:
            fh.write(json.dumps({"messages": e["messages"]}) + "\n")
    print(name, len(part))