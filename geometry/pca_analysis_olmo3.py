# -*- coding: utf-8 -*-
"""
PCA / Cosine Analysis of Cross-Lingual Steering Vectors
OLMo3-7B: EN / IT / KK / ZH / KO (Layers 12-16)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
LAYERS = list(range(12, 17))
LANGUAGES = ["en", "it", "kk", "zh", "ko"]

BASE = Path("/users/acp25mk/cross-lingual-caa/vectors/Olmo-3-7B-Instruct")
VECTOR_DIRS = {
    "en": BASE / "sycophancy/caa_vector_multiple_choice",
    "it": BASE / "sycophancy_it/sycophancy_it/caa_vector_multiple_choice",
    "kk": BASE / "sycophancy_kk/sycophancy_kk/caa_vector_multiple_choice",
    "zh": BASE / "sycophancy_zh/sycophancy_zh/caa_vector_multiple_choice",
    "ko": BASE / "sycophancy_ko/sycophancy_ko/caa_vector_multiple_choice",
}

OUTPUT_DIR = Path("/users/acp25mk/cross-lingual-caa/results/geometry_olmo3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
LANG_MARKERS = {"en": "o", "it": "s", "kk": "D", "zh": "^", "ko": "P"}
LANG_COLORS  = {"en": "#2196F3", "it": "#F44336", "kk": "#FF9800", "zh": "#4CAF50", "ko": "#9C27B0"}

def l2_normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)

def load_vector(lang, layer):
    path = VECTOR_DIRS[lang] / "layer_{}.pt".format(layer)
    if not path.exists():
        print("  MISSING: {}".format(path))
        return None
    return torch.load(path, weights_only=True).float().numpy()

def cosine(a, b):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))

def compute_pairwise_cosine():
    records = []
    for layer in LAYERS:
        vecs = {lang: load_vector(lang, layer) for lang in LANGUAGES}
        for i, l1 in enumerate(LANGUAGES):
            for l2 in LANGUAGES[i+1:]:
                if vecs[l1] is not None and vecs[l2] is not None:
                    records.append({
                        "layer": layer, "lang1": l1, "lang2": l2,
                        "cosine": cosine(vecs[l1], vecs[l2])
                    })
    return pd.DataFrame(records)

def plot_cosine_heatmap(df):
    mean_cos = np.zeros((len(LANGUAGES), len(LANGUAGES)))
    for i, l1 in enumerate(LANGUAGES):
        for j, l2 in enumerate(LANGUAGES):
            if i == j:
                mean_cos[i, j] = 1.0
            else:
                pair = df[((df.lang1==l1)&(df.lang2==l2)) | ((df.lang1==l2)&(df.lang2==l1))]
                mean_cos[i, j] = pair["cosine"].mean() if len(pair) else np.nan
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_cos, vmin=0.90, vmax=1.00, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = [l.upper() for l in LANGUAGES]
    ax.set_xticks(range(len(LANGUAGES))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(LANGUAGES))); ax.set_yticklabels(labels)
    for i in range(len(LANGUAGES)):
        for j in range(len(LANGUAGES)):
            if i != j:
                ax.text(j, i, "{:.3f}".format(mean_cos[i,j]), ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9)
    ax.set_title("Pairwise Cosine Similarity\n(mean across Layers, OLMo3-7B (Layers 12-16))")
    plt.tight_layout()
    out = OUTPUT_DIR / "cosine_heatmap_olmo3.png"
    plt.savefig(out, dpi=FIG_DPI); plt.close()
    print("Saved: {}".format(out))
    return mean_cos

def plot_norm():
    fig, ax = plt.subplots(figsize=(6, 4))
    for lang in LANGUAGES:
        norms = []
        for layer in LAYERS:
            v = load_vector(lang, layer)
            norms.append(np.linalg.norm(v) if v is not None else np.nan)
        ax.plot(LAYERS, norms, marker=LANG_MARKERS[lang],
                color=LANG_COLORS[lang], label=lang.upper())
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 Norm")
    ax.set_title("Vector L2 Norm by Language and Layer\nOLMo3-7B (Layers 12-16)")
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "norm_olmo3.png"
    plt.savefig(out, dpi=FIG_DPI); plt.close()
    print("Saved: {}".format(out))

def plot_en_target_cosine():
    fig, ax = plt.subplots(figsize=(6, 4))
    for lang in [l for l in LANGUAGES if l != "en"]:
        cosines = []
        for layer in LAYERS:
            en = load_vector("en", layer)
            tgt = load_vector(lang, layer)
            cosines.append(cosine(en, tgt) if (en is not None and tgt is not None) else np.nan)
        ax.plot(LAYERS, cosines, marker=LANG_MARKERS[lang],
                color=LANG_COLORS[lang], label=lang.upper())
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("EN-Target Cosine Similarity by Layer\nOLMo3-7B (Layers 12-16)")
    ax.set_ylim(0.25, 1.01)
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "en_target_cosine_olmo3.png"
    plt.savefig(out, dpi=FIG_DPI); plt.close()
    print("Saved: {}".format(out))

def print_summary(mean_cos):
    print("\n=== Mean EN-target cosine similarity (OLMo3, L12-16) ===")
    j = LANGUAGES.index("en")
    for i, lang in enumerate(LANGUAGES):
        if lang != "en":
            print("  EN-{}: {:.3f}".format(lang.upper(), mean_cos[j, i]))

if __name__ == "__main__":
    print("Computing pairwise cosine similarities...")
    df = compute_pairwise_cosine()
    df.to_csv(OUTPUT_DIR / "cosine_data_olmo3.csv", index=False)
    print("Plotting heatmap...")
    mean_cos = plot_cosine_heatmap(df)
    print("Plotting norms...")
    plot_norm()
    print("Plotting EN-target cosine by layer...")
    plot_en_target_cosine()
    print_summary(mean_cos)
    print("\nDone!")
