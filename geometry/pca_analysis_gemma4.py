"""
PCA / Cosine Analysis of Cross-Lingual Steering Vectors
Gemma4-12B-IT: EN / IT / KK / ZH / KO (Layers 24-28)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# ── Configuration ────────────────────────────────────────────
LAYERS = list(range(24, 29))
LANGUAGES = ["en", "it", "kk", "zh", "ko"]

VECTOR_DIRS = {
    "en": Path("/users/acp25mk/cross-lingual-caa/vectors/Gemma4-12B/sycophancy/caa_vector_multiple_choice"),
    "it": Path("/users/acp25mk/cross-lingual-caa/vectors/Gemma4-12B/sycophancy_it/caa_vector_multiple_choice"),
    "kk": Path("/users/acp25mk/cross-lingual-caa/vectors/Gemma4-12B/sycophancy_kk/caa_vector_multiple_choice"),
    "zh": Path("/users/acp25mk/cross-lingual-caa/vectors/Gemma4-12B/sycophancy_zh/caa_vector_multiple_choice"),
    "ko": Path("/users/acp25mk/cross-lingual-caa/vectors/Gemma4-12B/sycophancy_ko/caa_vector_multiple_choice"),
}

OUTPUT_DIR = Path("/users/acp25mk/cross-lingual-caa/results/geometry_gemma4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
RANDOM_STATE = 42
HIGH_COSINE_THRESHOLD = 0.8
LOW_COSINE_THRESHOLD  = 0.3

LANG_MARKERS = {"en": "o", "it": "s", "kk": "D", "zh": "^", "ko": "P"}
LANG_COLORS  = {"en": "#2196F3", "it": "#F44336", "kk": "#FF9800", "zh": "#4CAF50", "ko": "#9C27B0"}

# ── Utilities ────────────────────────────────────────────────
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

def load_vector(lang: str, layer: int) -> Optional[np.ndarray]:
    path = VECTOR_DIRS[lang] / f"layer_{layer}.pt"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    v = torch.load(path, map_location="cpu")
    if isinstance(v, torch.Tensor):
        return v.float().numpy()
    return None

def load_all_vectors() -> Dict[str, Dict[int, np.ndarray]]:
    vecs = {}
    for lang in LANGUAGES:
        vecs[lang] = {}
        for layer in LAYERS:
            v = load_vector(lang, layer)
            if v is not None:
                vecs[lang][layer] = v
    return vecs

# ── Norm Analysis ────────────────────────────────────────────
def analyze_norms(vecs):
    print("\n=== NORM ANALYSIS ===")
    rows = []
    for lang in LANGUAGES:
        for layer in LAYERS:
            if layer in vecs[lang]:
                norm = np.linalg.norm(vecs[lang][layer])
                rows.append({"lang": lang, "layer": layer, "norm": norm})
                print(f"  {lang} L{layer}: norm={norm:.4f}")
    df = pd.DataFrame(rows)
    print("\nMean norm per language:")
    print(df.groupby("lang")["norm"].mean().round(4))
    return df

# ── Cosine Similarity ────────────────────────────────────────
def analyze_cosine(vecs):
    print("\n=== COSINE SIMILARITY (EN vs others) ===")
    rows = []
    for layer in LAYERS:
        if layer not in vecs["en"]:
            continue
        en_v = l2_normalize(vecs["en"][layer])
        for lang in LANGUAGES:
            if lang == "en" or layer not in vecs[lang]:
                continue
            other_v = l2_normalize(vecs[lang][layer])
            cos = float(np.dot(en_v, other_v))
            rows.append({"lang_pair": f"en-{lang}", "layer": layer, "cosine": cos})
            print(f"  EN vs {lang.upper()} L{layer}: cosine={cos:.4f}")
    df = pd.DataFrame(rows)
    print("\nMean cosine per pair:")
    print(df.groupby("lang_pair")["cosine"].mean().round(4))
    return df

# ── Global PCA ────────────────────────────────────────────────
def run_global_pca(vecs):
    print("\n=== GLOBAL PCA ===")
    X, labels = [], []
    for lang in LANGUAGES:
        for layer in LAYERS:
            if layer in vecs[lang]:
                X.append(l2_normalize(vecs[lang][layer]))
                labels.append({"lang": lang, "layer": layer})
    X = np.array(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Total: {sum(pca.explained_variance_ratio_):.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(labels):
        lang, layer = lbl["lang"], lbl["layer"]
        ax.scatter(coords[i, 0], coords[i, 1],
                   marker=LANG_MARKERS[lang],
                   color=LANG_COLORS[lang],
                   s=60, alpha=0.8)
        ax.annotate(f"{lang[0].upper()}{layer}", (coords[i, 0], coords[i, 1]),
                    fontsize=6, alpha=0.7)

    for lang in LANGUAGES:
        ax.scatter([], [], marker=LANG_MARKERS[lang],
                   color=LANG_COLORS[lang], label=lang.upper())
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Global PCA — Qwen3.5-9B Sycophancy Steering Vectors\n(L2-normalised, layers 13–18)")
    plt.tight_layout()
    out = OUTPUT_DIR / "global_pca_gemma4.png"
    plt.savefig(out, dpi=FIG_DPI)
    print(f"  Saved: {out}")
    plt.close()

# ── Cosine Plot ───────────────────────────────────────────────
def plot_cosine(df_cos):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in df_cos["lang_pair"].unique():
        sub = df_cos[df_cos["lang_pair"] == pair].sort_values("layer")
        lang = pair.split("-")[1]
        ax.plot(sub["layer"], sub["cosine"],
                marker="o", color=LANG_COLORS[lang], label=f"EN vs {lang.upper()}")
    ax.axhline(HIGH_COSINE_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label="High threshold (0.8)")
    ax.axhline(LOW_COSINE_THRESHOLD, color="lightgray", linestyle="--", alpha=0.5, label="Low threshold (0.3)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("EN vs Other Languages — Cosine Similarity per Layer\nQwen3.5-9B Sycophancy Vectors")
    ax.legend()
    ax.set_xticks(LAYERS)
    plt.tight_layout()
    out = OUTPUT_DIR / "cosine_similarity_gemma4.png"
    plt.savefig(out, dpi=FIG_DPI)
    print(f"  Saved: {out}")
    plt.close()

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading vectors...")
    vecs = load_all_vectors()
    for lang in LANGUAGES:
        print(f"  {lang}: {len(vecs[lang])} layers loaded")

    df_norm = analyze_norms(vecs)
    df_cos = analyze_cosine(vecs)
    run_global_pca(vecs)
    plot_cosine(df_cos)

    # Save results
    df_norm.to_csv(OUTPUT_DIR / "norm_analysis_gemma4.csv", index=False)
    df_cos.to_csv(OUTPUT_DIR / "cosine_analysis_gemma4.csv", index=False)
    print(f"\nDone! Results saved to {OUTPUT_DIR}")
