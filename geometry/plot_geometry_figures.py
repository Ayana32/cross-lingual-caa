"""
Generate heatmap and norm line plot for both Qwen3.5 and Gemma4.
Run this script to regenerate geometry figures in the same style.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# ── Config ────────────────────────────────────────────────────
MODELS = {
    "qwen35": {
        "layers": list(range(13, 19)),
        "vector_dirs": {
            "en": Path("vectors/Qwen3.5-9B/sycophancy/caa_vector_multiple_choice"),
            "it": Path("vectors/Qwen3.5-9B-it/sycophancy_it/caa_vector_multiple_choice"),
            "kk": Path("vectors/Qwen3.5-9B-kk/sycophancy_kk/caa_vector_multiple_choice"),
            "zh": Path("vectors/Qwen3.5-9B-zh/sycophancy_zh/caa_vector_multiple_choice"),
            "ko": Path("vectors/Qwen3.5-9B-ko/sycophancy_ko/caa_vector_multiple_choice"),
        },
        "output_dir": Path("results/geometry_qwen35"),
        "title_suffix": "Qwen3.5-9B (Layers 13–18)",
    },
    "gemma4": {
        "layers": list(range(24, 29)),
        "vector_dirs": {
            "en": Path("vectors/Gemma4-12B/sycophancy/caa_vector_multiple_choice"),
            "it": Path("vectors/Gemma4-12B/sycophancy_it/caa_vector_multiple_choice"),
            "kk": Path("vectors/Gemma4-12B/sycophancy_kk/caa_vector_multiple_choice"),
            "zh": Path("vectors/Gemma4-12B/sycophancy_zh/caa_vector_multiple_choice"),
            "ko": Path("vectors/Gemma4-12B/sycophancy_ko/caa_vector_multiple_choice"),
        },
        "output_dir": Path("results/geometry_gemma4"),
        "title_suffix": "Gemma4-12B-IT (Layers 24–28)",
    },
}

LANGUAGES = ["en", "it", "zh", "ko", "kk"]
LANG_COLORS = {"en": "#1f77b4", "it": "#d62728", "zh": "#2ca02c", "ko": "#9467bd", "kk": "#ff7f0e"}
LANG_MARKERS = {"en": "o", "it": "s", "zh": "^", "ko": "D", "kk": "v"}
FIG_DPI = 150

def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def load_vectors(vector_dirs, layers):
    vecs = {}
    for lang in LANGUAGES:
        vecs[lang] = {}
        for layer in layers:
            path = vector_dirs[lang] / f"layer_{layer}.pt"
            if path.exists():
                v = torch.load(path, map_location="cpu")
                if isinstance(v, dict):
                    v = list(v.values())[0]
                vecs[lang][layer] = v.float().numpy()
    return vecs

def plot_heatmap(vecs, layers, title, save_path):
    """5x5 pairwise cosine similarity heatmap (mean across layers)."""
    langs = LANGUAGES
    matrix = np.full((len(langs), len(langs)), np.nan)
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i == j:
                continue
            cossims = []
            for layer in layers:
                if layer in vecs[l1] and layer in vecs[l2]:
                    v1 = l2_normalize(vecs[l1][layer])
                    v2 = l2_normalize(vecs[l2][layer])
                    cossims.append(float(np.dot(v1, v2)))
            if cossims:
                matrix[i][j] = np.mean(cossims)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = np.nanmin(matrix) - 0.01
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=1.0)
    plt.colorbar(im, ax=ax)
    labels = [l.upper() for l in langs]
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(langs)):
        for j in range(len(langs)):
            if not np.isnan(matrix[i][j]):
                text_color = "white" if matrix[i][j] > (vmin + (1.0 - vmin) * 0.75) else "black"
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center",
                       fontsize=9, color=text_color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_norm(vecs, layers, title, save_path):
    """L2 norm line plot by language and layer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for lang in LANGUAGES:
        norms = [np.linalg.norm(vecs[lang][l]) for l in layers if l in vecs[lang]]
        lyrs = [l for l in layers if l in vecs[lang]]
        ax.plot(lyrs, norms, marker=LANG_MARKERS[lang],
                color=LANG_COLORS[lang], linewidth=2, label=lang.upper())
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ── Main ──────────────────────────────────────────────────────
for model_name, cfg in MODELS.items():
    print(f"\n=== {model_name} ===")
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    vecs = load_vectors(cfg["vector_dirs"], cfg["layers"])

    plot_heatmap(
        vecs, cfg["layers"],
        f"Pairwise Cosine Similarity\n(mean across Layers, {cfg['title_suffix']})",
        cfg["output_dir"] / f"cosine_heatmap_{model_name}.png"
    )
    plot_norm(
        vecs, cfg["layers"],
        f"Vector L2 Norm by Language and Layer\n{cfg['title_suffix']}",
        cfg["output_dir"] / f"norm_{model_name}.png"
    )

print("\nAll figures saved!")
