# -*- coding: utf-8 -*-
"""
Cosine vs DeltaSR Scatter Plot - OLMo3-7B
Panel A: Between-language (peak layer)
Panel B: Within-language layer-wise (L12-16)
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

RESULTS_DIR = Path("results")
COSINE_CSV  = Path("results/geometry_olmo3/cosine_data_olmo3.csv")
OUTPUT_DIR  = Path("results/geometry_olmo3")
LAYERS      = list(range(12, 17))

LANGS = ["it", "zh", "ko", "kk"]
PEAK_LAYERS = {"it": 14, "zh": 14, "ko": 14, "kk": 14}

RESULT_DIRS = {
    "it": "olmo3_syco_it_m-1.5_layer{L}",
    "zh": "olmo3_syco_zh_m-1.5_layer{L}",
    "ko": "olmo3_syco_ko_m-1.5_layer{L}",
    "kk": "olmo3_syco_kk_m-1.5_layer{L}",
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

def compute_delta(result_path, source_path):
    results = json.load(open(result_path))
    sources = json.load(open(source_path))
    orig_syco, steer_syco, total = 0, 0, 0
    for res, src in zip(results, sources):
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
    return orig_syco/total*100, steer_syco/total*100, (steer_syco-orig_syco)/total*100

# Load cosine — lang1/lang2 형식을 en-target 형식으로 변환
df_cos_raw = pd.read_csv(COSINE_CSV)
cosine_lookup = {}
for _, row in df_cos_raw.iterrows():
    if row["lang1"] == "en":
        cosine_lookup[(row["lang2"], int(row["layer"]))] = row["cosine"]
    elif row["lang2"] == "en":
        cosine_lookup[(row["lang1"], int(row["layer"]))] = row["cosine"]

# Compute DeltaSR
records = []
for lang in LANGS:
    for layer in LAYERS:
        result_path = RESULTS_DIR / RESULT_DIRS[lang].format(L=layer) / "syco_{}_results.json".format(lang)
        source_path = Path(SOURCE_FILES[lang])
        if not result_path.exists():
            print("[SKIP] {}".format(result_path))
            continue
        orig_rate, steer_rate, delta = compute_delta(result_path, source_path)
        if delta is None:
            continue
        cos = cosine_lookup.get((lang, layer))
        if cos is None:
            continue
        records.append({
            "lang": lang, "layer": layer,
            "cosine": cos, "delta": delta,
            "abs_delta": abs(delta),
        })
        print("{} L{}: cosine={:.4f}, delta={:.1f}pp".format(lang, layer, cos, delta))

df = pd.DataFrame(records)
print("\nTotal records: {}".format(len(df)))

def panel_a(ax, df):
    for lang in LANGS:
        peak_L = PEAK_LAYERS[lang]
        row = df[(df["lang"] == lang) & (df["layer"] == peak_L)]
        if row.empty:
            continue
        x = row["cosine"].values[0]
        y = row["abs_delta"].values[0]
        ax.scatter(x, y, color=LANG_COLORS[lang], marker=LANG_MARKERS[lang], s=120, zorder=5)
        offsets = {"it": (5, 4), "zh": (5, 4), "ko": (5, -10), "kk": (5, 4)}
        ox, oy = offsets.get(lang, (5, 4))
        ax.annotate("{} (L{})".format(LANG_NAMES[lang], peak_L),
                    (x, y), fontsize=9, xytext=(ox, oy), textcoords="offset points")
    ax.set_xlabel("EN-target cosine similarity at L14", fontsize=10)
    ax.set_ylabel("|DeltaSR| (pp)", fontsize=10)
    ax.set_ylim(0, 25)
    ax.set_title("A. Fixed-layer steering effect and cosine similarity", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

def panel_b(ax, df):
    spearman_results = {}
    for lang in LANGS:
        sub = df[df["lang"] == lang].sort_values("layer")
        if sub.empty:
            continue
        xs = sub["cosine"].values
        ys = sub["abs_delta"].values
        ls = sub["layer"].values
        ax.plot(xs, ys, color=LANG_COLORS[lang], alpha=0.15, linewidth=0.8, zorder=2)
        for x, y, l in zip(xs, ys, ls):
            ax.scatter(x, y, color=LANG_COLORS[lang], marker=LANG_MARKERS[lang], s=80, zorder=5)
            ax.annotate(str(l), (x, y), fontsize=7, color=LANG_COLORS[lang],
                        xytext=(4, 4), textcoords="offset points")
        if len(xs) >= 4:
            rho, _ = spearmanr(xs, ys)
            spearman_results[lang] = rho
    handles = [
        mpatches.Patch(color=LANG_COLORS[l],
                       label="{} (rho = {:.2f})".format(LANG_NAMES[l], spearman_results.get(l, float("nan"))))
        for l in LANGS if l in spearman_results
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel("EN-target cosine similarity", fontsize=10)
    ax.set_ylabel("|DeltaSR| (pp)", fontsize=10)
    ax.set_title("B. Within-language, layer-wise (L12-16)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return spearman_results

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panel_a(axes[0], df)
rho_table = panel_b(axes[1], df)
plt.suptitle("Relationship between Cross-lingual Vector Similarity and Steering Effect\n(OLMo3-7B, Sycophancy)",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
out = OUTPUT_DIR / "cosine_delta_scatter_olmo3.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: {}".format(out))

print("\nWithin-language Spearman rho (descriptive, n=5):")
for lang, rho in rho_table.items():
    print("  {}: rho = {:.3f}".format(LANG_NAMES[lang], rho))

df.to_csv(OUTPUT_DIR / "cosine_delta_olmo3.csv", index=False)
