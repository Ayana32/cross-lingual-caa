# -*- coding: utf-8 -*-
"""
Cosine ↔ ΔSR Scatter Plot
Panel A: Between-language (peak layer, descriptive)
Panel B: Within-language layer-wise (L13-18, with Spearman ρ per language)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "cross_lingual"))
from parse_utils import parse_prediction

# ── Config ────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
COSINE_CSV  = Path("results/geometry_qwen35/cosine_analysis_qwen35.csv")
OUTPUT_DIR  = Path("results/geometry_qwen35")
LAYERS      = list(range(13, 19))

LANGS = ["it", "zh", "ko", "kk"]
PEAK_LAYERS = {"it": 15, "zh": 15, "ko": 15, "kk": 15}

RESULT_DIRS = {
    "it": "qwen35_syco_it_m-1.5_layer{L}",
    "zh": "qwen35_syco_zh_m-1.5_layer{L}",
    "ko": "qwen35_syco_ko_m-1.5_layer{L}",
    "kk": "qwen35_syco_kk_m-1.5_layer{L}",
}

SOURCE_FILES = {
    "it": "data/sycophancy/it/eval.json",
    "zh": "data/sycophancy/zh/eval.json",
    "ko": "data/sycophancy/ko/eval.json",
    "kk": "data/sycophancy/kk/eval.json",
}

LANG_COLORS  = {"it": "#E63946", "zh": "#2A9D8F", "ko": "#457B9D", "kk": "#E9C46A"}
LANG_MARKERS = {"it": "o", "zh": "s", "ko": "^", "kk": "D"}
LANG_NAMES   = {"it": "Italian", "zh": "Chinese", "ko": "Korean", "kk": "Kazakh"}

# ── Helpers ───────────────────────────────────────────────────
def compute_delta(result_path, source_path):
    """sycophancy rate delta 계산 (기존 analyze_sweep.py와 동일 로직)"""
    results = json.load(open(result_path))
    sources = json.load(open(source_path))

    orig_syco, steer_syco, total = 0, 0, 0
    for i, (res, src) in enumerate(zip(results, sources)):
        orig_pred  = parse_prediction(res.get("orig_pred", [""])[0])
        steer_pred = parse_prediction(res.get("pred", [""])[0])
        if orig_pred is None or steer_pred is None:
            continue
        ref = src.get("matching", "")
        orig_syco  += int(orig_pred  == ref)
        steer_syco += int(steer_pred == ref)
        total += 1

    if total == 0:
        return None, None, None
    orig_rate  = orig_syco  / total * 100
    steer_rate = steer_syco / total * 100
    delta      = steer_rate - orig_rate
    return orig_rate, steer_rate, delta

# ── Load cosine data ──────────────────────────────────────────
df_cos = pd.read_csv(COSINE_CSV)
df_cos["lang"] = df_cos["lang_pair"].str.replace("en-", "")

# ── Compute ΔSR per language per layer ───────────────────────
records = []
for lang in LANGS:
    for layer in LAYERS:
        result_dir  = RESULT_DIRS[lang].format(L=layer)
        result_path = RESULTS_DIR / result_dir / f"syco_{lang}_results.json"
        source_path = Path(SOURCE_FILES[lang])

        if not result_path.exists():
            print(f"[SKIP] {result_path} not found")
            continue

        orig_rate, steer_rate, delta = compute_delta(result_path, source_path)
        if delta is None:
            continue

        cos_row = df_cos[(df_cos["lang"] == lang) & (df_cos["layer"] == layer)]
        if cos_row.empty:
            continue
        cosine = cos_row["cosine"].values[0]

        records.append({
            "lang": lang, "layer": layer,
            "cosine": cosine, "delta": delta,
            "abs_delta": abs(delta),
            "orig_rate": orig_rate,
        })
        print(f"{lang} L{layer}: cosine={cosine:.4f}, delta={delta:.1f}pp")

df = pd.DataFrame(records)
print(f"\nTotal records: {len(df)}")
print(df)

# ── Panel A: Between-language peak scatter ────────────────────
def panel_a(ax, df):
    for lang in LANGS:
        peak_L = PEAK_LAYERS[lang]
        row = df[(df["lang"] == lang) & (df["layer"] == peak_L)]
        if row.empty:
            continue
        x = row["cosine"].values[0]
        y = row["abs_delta"].values[0]
        ax.scatter(x, y,
                   color=LANG_COLORS[lang],
                   marker=LANG_MARKERS[lang],
                   s=120, zorder=5)
        offsets = {"it": (-60, 4), "zh": (5, 4), "ko": (5, 4), "kk": (5, 4)}
        ox, oy = offsets.get(lang, (5, 4))
        ax.annotate(f"{LANG_NAMES[lang]} (L{peak_L})",
                    (x, y), fontsize=9,
                    xytext=(ox, oy), textcoords="offset points")

    ax.set_xlabel("EN–target cosine similarity at L15", fontsize=10)
    ax.set_ylabel("|ΔSR| (pp)", fontsize=10)
    ax.set_title("A. Fixed-layer steering effect and cosine similarity", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

# ── Panel B: Within-language layer-wise scatter ───────────────
def panel_b(ax, df):
    spearman_results = {}
    for lang in LANGS:
        sub = df[df["lang"] == lang].sort_values("layer")
        if sub.empty:
            continue
        xs = sub["cosine"].values
        ys = sub["abs_delta"].values
        ls = sub["layer"].values

        # 선으로 연결
        ax.plot(xs, ys,
                color=LANG_COLORS[lang],
                alpha=0.15, linewidth=0.8, zorder=2)

        # 점 + layer annotation
        for x, y, l in zip(xs, ys, ls):
            ax.scatter(x, y,
                       color=LANG_COLORS[lang],
                       marker=LANG_MARKERS[lang],
                       s=80, zorder=5)
            ax.annotate(str(l), (x, y),
                        fontsize=7, color=LANG_COLORS[lang],
                        xytext=(4, 4), textcoords="offset points")

        # within-language Spearman ρ
        if len(xs) >= 4:
            rho, _ = spearmanr(xs, ys)
            spearman_results[lang] = rho

    # legend
    handles = [
        mpatches.Patch(color=LANG_COLORS[l],
                       label=f"{LANG_NAMES[l]} (ρ = {spearman_results.get(l, float('nan')):.2f})")
        for l in LANGS if l in spearman_results
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel("EN–target cosine similarity", fontsize=10)
    ax.set_ylabel("|ΔSR| (pp)", fontsize=10)
    ax.set_title("B. Within-language, layer-wise (L13–18)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    return spearman_results

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panel_a(axes[0], df)
rho_table = panel_b(axes[1], df)

plt.suptitle("Relationship between Cross-lingual Vector Similarity and Steering Effect\n(Qwen3.5-9B, Sycophancy)",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()

out = OUTPUT_DIR / "cosine_delta_scatter_qwen35.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ── Spearman table ────────────────────────────────────────────
print("\nWithin-language Spearman ρ (descriptive, n=6):")
for lang, rho in rho_table.items():
    print(f"  {LANG_NAMES[lang]}: ρ = {rho:.3f}")

# Save data
df.to_csv(OUTPUT_DIR / "cosine_delta_qwen35.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/cosine_delta_qwen35.csv")
