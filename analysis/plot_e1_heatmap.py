#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LANGS = ["EN", "IT", "ZH", "KO", "KK"]
LANG_LOWER = ["en", "it", "zh", "ko", "kk"]

def delta_from_generation(result_dir, data_path):
    jsons = list(Path(result_dir).glob("*.json"))
    if not jsons:
        return None
    results = json.load(open(jsons[0]))
    data = json.load(open(data_path))
    n = len(results)
    baseline = sum(1 for r, d in zip(results, data)
                   if r['orig_pred'] and d['matching'] in r['orig_pred'][0])
    steered  = sum(1 for r, d in zip(results, data)
                   if r['pred'] and d['matching'] in r['pred'][0])
    return (steered - baseline) / n * 100

def delta_from_logprob(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.load(open(p))['delta_pp']

def build_qwen():
    M = np.full((5,5), np.nan)
    for i, src in enumerate(LANG_LOWER):
        for j, tgt in enumerate(LANG_LOWER):
            if i == j and src != 'en':
                M[i,j] = delta_from_generation(
                    f"results/qwen35_native_{src}_m-1.5_layer15",
                    f"data/sycophancy/{src}/eval.json")
            else:
                M[i,j] = delta_from_logprob(
                    f"results/qwen35_e1_{src}vec_{tgt}eval_m-1.5_layer15/results.json")
    return M

def build_gemma4():
    M = np.full((5,5), np.nan)
    for i, src in enumerate(LANG_LOWER):
        for j, tgt in enumerate(LANG_LOWER):
            if i == j:
                rd = (f"results/gemma4_syco_en_m-1.0_layer26" if src == 'en'
                      else f"results/gemma4_native_{src}_m-1.0_layer26")
                M[i,j] = delta_from_generation(rd, f"data/sycophancy/{src}/eval.json")
            else:
                M[i,j] = delta_from_generation(
                    f"results/gemma4_e1_{src}vec_{tgt}eval_m-1.0_layer26",
                    f"data/sycophancy/{tgt}/eval.json")
    return M

qwen_M   = build_qwen()
gemma4_M = build_gemma4()

print("Qwen3.5 ΔSR matrix:")
print(np.round(qwen_M, 1))
print("\nGemma4 ΔSR matrix:")
print(np.round(gemma4_M, 1))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
vmin, vmax = -40, 10

for ax, M, title in zip(axes,
                         [qwen_M, gemma4_M],
                         ["Qwen3.5-9B  (L15, m=-1.5)",
                          "Gemma4-12B  (L26, m=-1.0)"]):
    im = ax.imshow(M, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels(LANGS)
    ax.set_yticks(range(5)); ax.set_yticklabels(LANGS)
    ax.set_xlabel("Eval language"); ax.set_ylabel("Vector language")
    ax.set_title(f"ΔSR (pp)\n{title}", fontsize=11)
    for i in range(5):
        for j in range(5):
            val = M[i,j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                        fontsize=9, color='black' if abs(val) < 20 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("5×5 Cross-lingual Sycophancy Transfer Matrix", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/e1_heatmap_both.png", dpi=150, bbox_inches='tight')
print("\nSaved: results/e1_heatmap_both.png")
