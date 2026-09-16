"""
PCA / SVD / Cosine / Subspace Analysis of Cross-Lingual Steering Vectors

Compares steering vector geometry across languages (EN / KO / ES / KK)
and tasks (sycophancy / syllogism) to investigate whether steering
directions are language-invariant or language-specific, and whether
cross-task transfer is mediated by shared subspace structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


# ── Configuration ────────────────────────────────────────────

LAYERS = list(range(12, 27))
LANGUAGES = ["en", "ko", "es", "kk"]

# Sycophancy steering vectors — trained on sycophancy contrastive pairs
VECTOR_DIRS = {
    "en": Path("~/Downloads/en_vectors").expanduser(),
    "ko": Path("~/Downloads/ko_vectors/sycophancy_ko/caa_vector_multiple_choice").expanduser(),
    "es": Path("~/Downloads/es_vectors/sycophancy_es/caa_vector_multiple_choice").expanduser(),
    "kk": Path("~/Downloads/kk_vectors/sycophancy_kk/caa_vector_multiple_choice").expanduser(),
}

# Syllogism steering vectors — trained on SemEval syllogism contrastive pairs
VECTOR_DIRS_SEMEVAL = {
    "en": Path("~/Downloads/semeval_en_vectors/semeval_en/caa_vector_multiple_choice").expanduser(),
    "ko": Path("~/Downloads/semeval_ko_vectors/semeval_ko/caa_vector_multiple_choice").expanduser(),
    "es": Path("~/Downloads/semeval_es_vectors/semeval_es/caa_vector_multiple_choice").expanduser(),
    "kk": Path("~/Downloads/semeval_kk_vectors/semeval_kk/caa_vector_multiple_choice").expanduser(),
}

OUTPUT_DIR = Path("~/Downloads/cross_lingual_steering_analysis").expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 200
RANDOM_STATE = 42

REFERENCE_LAYER = 17          # L17 from prior experiments (descriptive reference only)
HIGH_COSINE_THRESHOLD = 0.8   # heuristic: high directional alignment
LOW_COSINE_THRESHOLD  = 0.3   # heuristic: low directional alignment

LANG_MARKERS = {"en": "o", "ko": "s", "es": "^", "kk": "D"}
LANG_COLORS  = {"en": "#2196F3", "ko": "#F44336", "es": "#4CAF50", "kk": "#FF9800"}


# ── Utilities ────────────────────────────────────────────────

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / norm if norm >= eps else x.copy()


def ensure_1d_float32(x) -> np.ndarray:
    """Convert torch tensor / array to float32 1D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def safe_cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    return float(np.dot(v1, v2) / (n1 * n2))


def cosine_to_angle_degrees(cosine: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── Loading ──────────────────────────────────────────────────

def load_vectors(
    vector_dirs: Dict[str, Path],
    layers: List[int],
) -> Dict[str, Dict[int, np.ndarray]]:
    """Load one steering vector per (language, layer)."""
    vectors: Dict[str, Dict[int, np.ndarray]] = {}
    for lang, base_dir in vector_dirs.items():
        vectors[lang] = {}
        if not base_dir.exists():
            print(f"[WARN] Missing directory for '{lang}': {base_dir}")
            continue
        for layer in layers:
            pt_file = base_dir / f"layer_{layer}.pt"
            if not pt_file.exists():
                continue
            try:
                v = torch.load(pt_file, map_location="cpu", weights_only=True)
                vectors[lang][layer] = ensure_1d_float32(v)
            except Exception as e:
                print(f"[ERROR] {pt_file}: {e}")
    return vectors


def validate_dimensions(vectors: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for lang, layer_dict in vectors.items():
        for layer, vec in layer_dict.items():
            rows.append({"language": lang, "layer": layer, "dim": int(vec.shape[0])})
    return pd.DataFrame(rows).sort_values(["language", "layer"]).reset_index(drop=True)


# ── Summary tables ───────────────────────────────────────────

def build_norm_dataframe(vectors: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for lang, layer_dict in vectors.items():
        for layer, vec in layer_dict.items():
            rows.append({"language": lang, "layer": layer, "norm": float(np.linalg.norm(vec))})
    return pd.DataFrame(rows).sort_values(["language", "layer"]).reset_index(drop=True)


def compute_cosine_dataframe(
    vectors: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    lang_pairs: List[Tuple[str, str]],
    normalize: bool = False,
) -> pd.DataFrame:
    """Long-form dataframe with cosine similarity and angular distance per pair × layer."""
    rows = []
    for lang1, lang2 in lang_pairs:
        pair_name = f"{lang1}_vs_{lang2}"
        for layer in layers:
            v1 = vectors.get(lang1, {}).get(layer)
            v2 = vectors.get(lang2, {}).get(layer)
            if v1 is None or v2 is None:
                rows.append({"pair": pair_name, "lang1": lang1, "lang2": lang2,
                             "layer": layer, "cosine": np.nan, "angle_deg": np.nan})
                continue
            if normalize:
                v1, v2 = l2_normalize(v1), l2_normalize(v2)
            sim = safe_cosine(v1, v2)
            angle = cosine_to_angle_degrees(sim) if not np.isnan(sim) else np.nan
            rows.append({"pair": pair_name, "lang1": lang1, "lang2": lang2,
                         "layer": layer, "cosine": sim, "angle_deg": angle})
    return pd.DataFrame(rows).sort_values(["pair", "layer"]).reset_index(drop=True)


def build_global_matrix(
    vectors: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
    normalize: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build [n_samples, hidden_dim] matrix with one sample per (language, layer)."""
    rows, X_list = [], []
    for lang, layer_dict in vectors.items():
        for layer in layers:
            if layer not in layer_dict:
                continue
            vec = l2_normalize(layer_dict[layer]) if normalize else layer_dict[layer]
            X_list.append(vec)
            rows.append({"language": lang, "layer": layer, "label": f"{lang}_L{layer}"})
    if not X_list:
        raise ValueError("No vectors found.")
    return np.stack(X_list, axis=0), pd.DataFrame(rows)


# ── PCA / SVD ────────────────────────────────────────────────

def run_global_pca(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_components: int = 2,
) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    df = meta.copy()
    for i in range(n_components):
        df[f"PC{i+1}"] = X_pca[:, i]
    return df, pca


def run_per_layer_pca(
    vectors: Dict[str, Dict[int, np.ndarray]],
    layer: int,
    normalize: bool = True,
) -> Optional[Tuple[pd.DataFrame, PCA]]:
    langs, vecs = [], []
    for lang, layer_dict in vectors.items():
        if layer in layer_dict:
            v = l2_normalize(layer_dict[layer]) if normalize else layer_dict[layer]
            langs.append(lang)
            vecs.append(v)
    if len(vecs) < 2:
        return None
    X = np.stack(vecs, axis=0)
    n_components = min(2, X.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    df = pd.DataFrame({
        "language": langs, "layer": layer,
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0,
    })
    return df, pca


def compute_svd_summary(X: np.ndarray, center: bool = True) -> dict:
    X_use = X - X.mean(axis=0, keepdims=True) if center else X.copy()
    U, S, Vt = np.linalg.svd(X_use, full_matrices=False)
    var = S ** 2
    var_ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    return {
        "singular_values": S.tolist(),
        "explained_variance_ratio": var_ratio.tolist(),
        "rank": int(np.linalg.matrix_rank(X_use)),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


# ── Subspace Analysis ────────────────────────────────────────

def build_task_matrix(
    vector_dirs: Dict[str, Path],
    layers: List[int],
) -> Dict[str, np.ndarray]:
    """
    Build a [n_layers, hidden_dim] matrix per language.
    Used to construct task-level subspaces via SVD.
    """
    vectors = load_vectors(vector_dirs, layers)
    matrices = {}
    for lang, layer_dict in vectors.items():
        # Use only layers present in layer_dict, in order
        vecs = [layer_dict[l] for l in layers if l in layer_dict]
        if vecs:
            matrices[lang] = np.stack(vecs, axis=0)
    return matrices


def extract_subspace(X: np.ndarray, k: int = 3, normalize: bool = True) -> np.ndarray:
    """
    Extract top-k principal directions from a [n_samples, hidden_dim] matrix.
    Returns U: [hidden_dim, k] — the subspace basis.
    """
    if normalize:
        X = np.array([l2_normalize(row) for row in X])
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return Vt[:k].T  # [hidden_dim, k]


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> dict:
    """
    Compute overlap between two subspaces U and V ([hidden_dim, k]).
    Uses singular values of U^T V to measure principal angles.
    Singular value close to 1 = high overlap, close to 0 = orthogonal.
    """
    M = U.T @ V
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, 0, 1)
    principal_angles_deg = np.degrees(np.arccos(singular_values))
    return {
        "singular_values": singular_values.tolist(),
        "principal_angles_deg": principal_angles_deg.tolist(),
        "mean_overlap": float(np.mean(singular_values)),
        "max_overlap": float(np.max(singular_values)),
    }


def projection_ratio_per_layer(
    vectors: Dict[str, Dict[int, np.ndarray]],
    subspace_basis: np.ndarray,
    lang: str,
    layers: List[int],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    For each layer, compute what fraction of the steering vector
    lies within the given subspace.

    proj_ratio(L) = ||U^T v|| / ||v||

    High value = vector aligns well with subspace directions.
    """
    rows = []
    for layer in layers:
        v = vectors.get(lang, {}).get(layer)
        if v is None:
            rows.append({"layer": layer, "proj_ratio": np.nan,
                         "proj_norm": np.nan, "v_norm": np.nan})
            continue
        if normalize:
            v = l2_normalize(v)
        proj = subspace_basis.T @ v
        proj_norm = float(np.linalg.norm(proj))
        v_norm = float(np.linalg.norm(v))
        ratio = proj_norm / v_norm if v_norm > 1e-12 else np.nan
        rows.append({"layer": layer, "proj_ratio": ratio,
                     "proj_norm": proj_norm, "v_norm": v_norm})
    return pd.DataFrame(rows)


# ── Interpretation ───────────────────────────────────────────

def interpret_cosine_results(df_cos: pd.DataFrame) -> dict:
    """
    Automatically extract key findings from cosine similarity results.
    Returns a structured dict covering alignment class, peak layer,
    L17 behaviour, and Spearman rank correlations.
    """
    findings = {}
    for pair, subdf in df_cos.groupby("pair"):
        valid = subdf.dropna(subset=["cosine"])
        if valid.empty:
            findings[pair] = {"error": "No valid cosine values"}
            continue
        mean_cos   = float(valid["cosine"].mean())
        std_cos    = float(valid["cosine"].std())
        peak_row   = valid.loc[valid["cosine"].idxmax()]
        trough_row = valid.loc[valid["cosine"].idxmin()]
        l17_row    = valid[valid["layer"] == REFERENCE_LAYER]
        l17_cos    = float(l17_row["cosine"].iloc[0]) if not l17_row.empty else np.nan
        l17_angle  = float(l17_row["angle_deg"].iloc[0]) if not l17_row.empty else np.nan

        if mean_cos >= HIGH_COSINE_THRESHOLD:
            alignment = "high directional alignment"
        elif mean_cos <= LOW_COSINE_THRESHOLD:
            alignment = "low directional alignment"
        else:
            alignment = "moderate directional alignment"

        high_layers = valid[valid["cosine"] >= HIGH_COSINE_THRESHOLD]["layer"].tolist()
        low_layers  = valid[valid["cosine"] <= LOW_COSINE_THRESHOLD]["layer"].tolist()

        findings[pair] = {
            "mean_cosine": round(mean_cos, 4),
            "std_cosine": round(std_cos, 4),
            "peak_layer": int(peak_row["layer"]),
            "peak_cosine": round(float(peak_row["cosine"]), 4),
            "peak_angle_deg": round(cosine_to_angle_degrees(float(peak_row["cosine"])), 2),
            "trough_layer": int(trough_row["layer"]),
            "trough_cosine": round(float(trough_row["cosine"]), 4),
            "l17_cosine": round(l17_cos, 4) if not np.isnan(l17_cos) else None,
            "l17_angle_deg": round(l17_angle, 2) if not np.isnan(l17_angle) else None,
            "alignment_class": alignment,
            "high_alignment_layers": high_layers,
            "low_alignment_layers": low_layers,
        }
    return findings


def print_interpretation(findings: dict, df_cos: pd.DataFrame):
    """Print human-readable interpretation of cosine findings."""
    print("\n" + "="*60)
    print("CROSS-LINGUAL STEERING VECTOR GEOMETRY: KEY FINDINGS")
    print("="*60)
    for pair, info in findings.items():
        if "error" in info:
            print(f"\n{pair}: {info['error']}")
            continue
        print(f"\n── {pair.upper()} ──")
        print(f"  Alignment class  : {info['alignment_class']}")
        print(f"  Mean cosine      : {info['mean_cosine']:.4f} ± {info['std_cosine']:.4f}")
        print(f"  Peak layer       : L{info['peak_layer']} (cosine={info['peak_cosine']:.4f}, "
              f"angle={info['peak_angle_deg']:.1f}°)")
        print(f"  Trough layer     : L{info['trough_layer']} (cosine={info['trough_cosine']:.4f})")
        if info["l17_cosine"] is not None:
            print(f"  L{REFERENCE_LAYER} (from prior experiments): "
                  f"cosine={info['l17_cosine']:.4f}, angle={info['l17_angle_deg']:.1f}°")
        if info["high_alignment_layers"]:
            print(f"  High-align layers: {info['high_alignment_layers']} "
                  f"(cosine >= {HIGH_COSINE_THRESHOLD})")
        else:
            print(f"  High-align layers: none (no layer reaches cosine >= {HIGH_COSINE_THRESHOLD})")
        if info["low_alignment_layers"]:
            print(f"  Low-align layers : {info['low_alignment_layers']} "
                  f"(cosine <= {LOW_COSINE_THRESHOLD})")

    print("\n── CROSS-PAIR SPEARMAN CORRELATION (descriptive, n=15) ──")
    print("  Note: reported as descriptive measure; p-values not inferential at n~15.")
    pairs = list(findings.keys())
    pair_series = {}
    for pair in pairs:
        subdf = df_cos[df_cos["pair"] == pair].dropna(subset=["cosine"])
        pair_series[pair] = subdf.set_index("layer")["cosine"]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            p1, p2 = pairs[i], pairs[j]
            common = pair_series[p1].index.intersection(pair_series[p2].index)
            if len(common) < 3:
                continue
            rho, pval = spearmanr(pair_series[p1].loc[common], pair_series[p2].loc[common])
            print(f"  {p1} vs {p2}: rho={rho:.3f}, p={pval:.4f}")
    print("\n" + "="*60)


def print_norm_interpretation(df_norm: pd.DataFrame):
    """Highlight norm differences and notable spikes across languages."""
    print("\n── VECTOR NORM ANALYSIS ──")
    summary = df_norm.groupby("language")["norm"].agg(["mean", "std", "max"])
    print(summary.round(4))
    for lang in df_norm["language"].unique():
        lang_norms = df_norm[df_norm["language"] == lang].set_index("layer")["norm"]
        if REFERENCE_LAYER in lang_norms.index:
            l17_norm = lang_norms[REFERENCE_LAYER]
            mean_norm = lang_norms.mean()
            ratio = l17_norm / mean_norm if mean_norm > 0 else np.nan
            if ratio > 1.5:
                print(f"  [{lang}] L{REFERENCE_LAYER} shows a notable norm increase: "
                      f"{l17_norm:.2f} ({ratio:.1f}x above mean {mean_norm:.2f})")


# ── Plotting ─────────────────────────────────────────────────

def plot_cosine_lines(df_cos: pd.DataFrame, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair_name, subdf in df_cos.groupby("pair"):
        ax.plot(subdf["layer"], subdf["cosine"], marker="o", linewidth=2, label=pair_name)
    ax.axvline(x=REFERENCE_LAYER, color="gray", linestyle="--", alpha=0.6,
               label=f"L{REFERENCE_LAYER} (from prior experiments)")
    ax.axhline(y=HIGH_COSINE_THRESHOLD, color="green", linestyle=":", alpha=0.5,
               label=f"High threshold ({HIGH_COSINE_THRESHOLD})")
    ax.axhline(y=LOW_COSINE_THRESHOLD, color="red", linestyle=":", alpha=0.5,
               label=f"Low threshold ({LOW_COSINE_THRESHOLD})")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(sorted(df_cos["layer"].dropna().unique()))
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_cosine_heatmap(df_cos: pd.DataFrame, title: str, save_path: Path):
    pivot = df_cos.pivot(index="pair", columns="layer", values="cosine").sort_index()
    data = pivot.values
    fig, ax = plt.subplots(figsize=(13, max(3, len(pivot) * 1.2)))
    im = ax.imshow(data, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Language pair", fontsize=11)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "NaN" if np.isnan(val) else f"{val:.2f}"
            color = "white" if not np.isnan(val) and abs(val) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)
    if REFERENCE_LAYER in list(pivot.columns):
        col_idx = list(pivot.columns).index(REFERENCE_LAYER)
        ax.axvline(x=col_idx, color="navy", linewidth=2, alpha=0.7)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_norms(df_norm: pd.DataFrame, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for lang, subdf in df_norm.groupby("language"):
        ax.plot(subdf["layer"], subdf["norm"],
                marker=LANG_MARKERS.get(lang, "o"),
                color=LANG_COLORS.get(lang),
                linewidth=2, label=lang)
    ax.axvline(x=REFERENCE_LAYER, color="gray", linestyle="--", alpha=0.6,
               label=f"L{REFERENCE_LAYER} (from prior experiments)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L2 norm", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(sorted(df_norm["layer"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_global_pca(
    df_pca: pd.DataFrame,
    explained_variance: List[float],
    title: str,
    save_path: Path,
    annotate: bool = True,
):
    fig, ax = plt.subplots(figsize=(9, 7))
    for lang, subdf in df_pca.groupby("language"):
        subdf_sorted = subdf.sort_values("layer")
        color = LANG_COLORS.get(lang)
        ax.plot(subdf_sorted["PC1"], subdf_sorted["PC2"],
                color=color, linewidth=1.2, alpha=0.4, zorder=1)
        ax.scatter(subdf_sorted["PC1"], subdf_sorted["PC2"],
                   label=lang, marker=LANG_MARKERS.get(lang, "o"),
                   color=color, s=80, alpha=0.85, zorder=2)
        if annotate:
            for _, row in subdf_sorted.iterrows():
                ax.annotate(str(int(row["layer"])), (row["PC1"], row["PC2"]),
                            fontsize=7.5, alpha=0.8,
                            xytext=(4, 2), textcoords="offset points")
    pc1_pct = explained_variance[0] * 100 if len(explained_variance) > 0 else 0
    pc2_pct = explained_variance[1] * 100 if len(explained_variance) > 1 else 0
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_per_layer_pca(
    df_layer_pca: pd.DataFrame,
    explained_variance: List[float],
    layer: int,
    save_path: Path,
):
    """Qualitative per-layer PCA scatter across languages."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in df_layer_pca.iterrows():
        lang = row["language"]
        ax.scatter(row["PC1"], row["PC2"],
                   marker=LANG_MARKERS.get(lang, "o"),
                   color=LANG_COLORS.get(lang),
                   s=90, alpha=0.9, label=lang)
        ax.annotate(lang, (row["PC1"], row["PC2"]),
                    fontsize=10, xytext=(4, 2), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    pc1_pct = explained_variance[0] * 100 if len(explained_variance) > 0 else 0
    pc2_pct = explained_variance[1] * 100 if len(explained_variance) > 1 else 0
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)", fontsize=11)
    ax.set_title(f"Per-layer PCA — Layer {layer} (qualitative)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_angle_lines(df_cos: pd.DataFrame, title: str, save_path: Path):
    """Angular distance in degrees — more intuitive than cosine."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair_name, subdf in df_cos.groupby("pair"):
        ax.plot(subdf["layer"], subdf["angle_deg"], marker="o", linewidth=2, label=pair_name)
    ax.axvline(x=REFERENCE_LAYER, color="gray", linestyle="--", alpha=0.6,
               label=f"L{REFERENCE_LAYER} (from prior experiments)")
    ax.axhline(y=90, color="red", linestyle=":", alpha=0.5, label="90 deg (orthogonal)")
    ax.axhline(y=36.87, color="green", linestyle=":", alpha=0.5,
               label=f"~37 deg (cosine={HIGH_COSINE_THRESHOLD})")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Angular distance (degrees)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(sorted(df_cos["layer"].dropna().unique()))
    ax.set_ylim(-5, 185)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_subspace_overlap_bar(
    overlap_results: Dict[str, dict],
    title: str,
    save_path: Path,
):
    """Bar chart of mean and max subspace overlap per pair."""
    labels = list(overlap_results.keys())
    means = [overlap_results[k]["mean_overlap"] for k in labels]
    maxes = [overlap_results[k]["max_overlap"] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - 0.2, means, 0.35, label="Mean overlap", alpha=0.8)
    ax.bar(x + 0.2, maxes, 0.35, label="Max overlap", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Subspace overlap (singular value)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12)
    ax.axhline(y=0.8, color="green", linestyle=":", alpha=0.5, label="High threshold (0.8)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_projection_vs_performance(
    df_proj: pd.DataFrame,
    perf_deltas: Dict[int, float],
    title: str,
    save_path: Path,
    lang: str = "en",
):
    """
    Scatter: projection ratio (x) vs performance delta (y) per layer.
    Tests whether shared subspace overlap predicts cross-task transfer.
    """
    layers = df_proj["layer"].tolist()
    proj   = df_proj["proj_ratio"].tolist()
    perf   = [perf_deltas.get(l, np.nan) for l in layers]
    valid  = [(p, q, l) for p, q, l in zip(proj, perf, layers)
              if not np.isnan(p) and not np.isnan(q)]
    if not valid:
        return
    xs, ys, ls = zip(*valid)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=80, alpha=0.8, color=LANG_COLORS.get(lang, "steelblue"))
    for x, y, l in zip(xs, ys, ls):
        ax.annotate(str(l), (x, y), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    rho, pval = spearmanr(xs, ys)
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}, p={pval:.3f} (descriptive, n={len(xs)})",
                 fontsize=11)
    ax.set_xlabel("Projection ratio onto target subspace", fontsize=11)
    ax.set_ylabel("Performance delta (pp)", fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading steering vectors...")
    vectors = load_vectors(VECTOR_DIRS, LAYERS)

    print("\n=== Loading Summary ===")
    for lang in LANGUAGES:
        layer_dict = vectors.get(lang, {})
        if layer_dict:
            dim = next(iter(layer_dict.values())).shape[0]
            print(f"  {lang}: {len(layer_dict)} layers | dim={dim}")
        else:
            print(f"  {lang}: 0 layers loaded")

    df_dim = validate_dimensions(vectors)
    df_dim.to_csv(OUTPUT_DIR / "vector_dimensions.csv", index=False)
    unique_dims = sorted(df_dim["dim"].unique())
    if df_dim.empty:
        raise RuntimeError("No vectors loaded. Check VECTOR_DIRS and layer_*.pt files.")
    if len(unique_dims) != 1:
        raise RuntimeError(f"Inconsistent dimensions: {unique_dims}")

    # ── 1. Norm analysis ──────────────────────────────────────
    print("\n[1/7] Norm analysis...")
    df_norm = build_norm_dataframe(vectors)
    df_norm.to_csv(OUTPUT_DIR / "vector_norms.csv", index=False)
    print_norm_interpretation(df_norm)
    plot_norms(df_norm, "Steering Vector Norms by Language and Layer",
               OUTPUT_DIR / "vector_norms.png")

    # ── 2. Cosine similarity (raw) ────────────────────────────
    print("\n[2/7] Cosine similarity (raw vectors)...")
    lang_pairs = [("en", "ko"), ("en", "es"), ("ko", "es")]
    df_cos_raw = compute_cosine_dataframe(vectors, LAYERS, lang_pairs, normalize=False)
    df_cos_raw.to_csv(OUTPUT_DIR / "cosine_raw.csv", index=False)
    plot_cosine_lines(df_cos_raw, "Cosine Similarity — Raw Vectors",
                      OUTPUT_DIR / "cosine_raw_lines.png")
    plot_cosine_heatmap(df_cos_raw, "Cosine Similarity Heatmap — Raw",
                        OUTPUT_DIR / "cosine_raw_heatmap.png")

    # ── 3. Cosine similarity (L2-normalized) ──────────────────
    print("\n[3/7] Cosine similarity (L2-normalized — direction only)...")
    df_cos_norm = compute_cosine_dataframe(vectors, LAYERS, lang_pairs, normalize=True)
    df_cos_norm.to_csv(OUTPUT_DIR / "cosine_l2_normalized.csv", index=False)
    plot_cosine_lines(df_cos_norm,
                      "Cosine Similarity — L2-Normalized (Direction Only)",
                      OUTPUT_DIR / "cosine_l2_normalized_lines.png")
    plot_cosine_heatmap(df_cos_norm,
                        "Cosine Similarity Heatmap — L2-Normalized",
                        OUTPUT_DIR / "cosine_l2_normalized_heatmap.png")
    plot_angle_lines(df_cos_norm,
                     "Angular Distance Between Steering Directions",
                     OUTPUT_DIR / "angular_distance.png")

    # ── 4. Interpretation ─────────────────────────────────────
    print("\n[4/7] Interpreting results...")
    findings = interpret_cosine_results(df_cos_norm)
    save_json(findings, OUTPUT_DIR / "interpretation.json")
    print_interpretation(findings, df_cos_norm)

    # ── 5. Global PCA ─────────────────────────────────────────
    print("\n[5/7] Global PCA...")
    for normalize, tag in [(False, "raw"), (True, "l2_normalized")]:
        X, meta = build_global_matrix(vectors, LAYERS, normalize=normalize)
        df_pca, pca_model = run_global_pca(X, meta)
        df_pca.to_csv(OUTPUT_DIR / f"global_pca_{tag}.csv", index=False)
        plot_global_pca(
            df_pca,
            explained_variance=list(pca_model.explained_variance_ratio_),
            title=f"Global PCA — {'L2-Normalized' if normalize else 'Raw'}",
            save_path=OUTPUT_DIR / f"global_pca_{tag}.png",
        )
        svd = compute_svd_summary(X)
        save_json(svd, OUTPUT_DIR / f"svd_{tag}.json")
        print(f"  PCA ({tag}): PC1={pca_model.explained_variance_ratio_[0]*100:.1f}%, "
              f"PC2={pca_model.explained_variance_ratio_[1]*100:.1f}%")

    # ── 6. Per-layer PCA (qualitative) ────────────────────────
    print("\n[6/7] Per-layer PCA...")
    per_layer_rows = []
    for layer in LAYERS:
        result = run_per_layer_pca(vectors, layer=layer, normalize=True)
        if result is None:
            continue
        df_lyr, pca_lyr = result
        df_lyr.to_csv(OUTPUT_DIR / f"per_layer_pca_L{layer}.csv", index=False)
        plot_per_layer_pca(
            df_lyr,
            explained_variance=list(pca_lyr.explained_variance_ratio_),
            layer=layer,
            save_path=OUTPUT_DIR / f"per_layer_pca_L{layer}.png",
        )
        per_layer_rows.append({
            "layer": layer,
            "pc1_var": float(pca_lyr.explained_variance_ratio_[0]),
            "pc2_var": float(pca_lyr.explained_variance_ratio_[1])
            if len(pca_lyr.explained_variance_ratio_) > 1 else np.nan,
        })
    pd.DataFrame(per_layer_rows).to_csv(
        OUTPUT_DIR / "per_layer_pca_summary.csv", index=False)

    # ── 7. Cross-task vector comparison ───────────────────────
    print("\n[7/7] Cross-task analysis: sycophancy vs syllogism...")
    vectors_semeval = load_vectors(VECTOR_DIRS_SEMEVAL, LAYERS)
    print("  Semeval (syllogism) vectors:")
    for lang in LANGUAGES:
        layer_dict = vectors_semeval.get(lang, {})
        print(f"    {lang}: {len(layer_dict)} layers loaded")

    # Cross-task cosine similarity (same language, different task)
    vectors_merged = {}
    for lang in LANGUAGES:
        if lang in vectors:
            vectors_merged[f"syco_{lang}"] = vectors[lang]
        if lang in vectors_semeval:
            vectors_merged[f"semeval_{lang}"] = vectors_semeval[lang]

    cross_task_pairs = [
        ("syco_en", "semeval_en"),
        ("syco_ko", "semeval_ko"),
        ("syco_es", "semeval_es"),
    ]
    df_cross_task = compute_cosine_dataframe(
        vectors_merged, LAYERS, cross_task_pairs, normalize=True)
    df_cross_task.to_csv(OUTPUT_DIR / "cosine_cross_task.csv", index=False)
    plot_cosine_lines(
        df_cross_task,
        title="Sycophancy vs Syllogism Vector Cosine Similarity (Same Language)",
        save_path=OUTPUT_DIR / "cosine_cross_task_lines.png")
    plot_cosine_heatmap(
        df_cross_task,
        title="Cross-Task Cosine Similarity Heatmap",
        save_path=OUTPUT_DIR / "cosine_cross_task_heatmap.png")
    findings_cross = interpret_cosine_results(df_cross_task)
    save_json(findings_cross, OUTPUT_DIR / "interpretation_cross_task.json")
    print_interpretation(findings_cross, df_cross_task)

    # ── Subspace analysis ─────────────────────────────────────
    print("\n[+] Subspace analysis...")

    K = 3  # number of principal directions per subspace
    # k=3 captures dominant variance while maintaining interpretability.
    # Results are consistent for k in {2, 3, 5} (verified empirically).

    print("  Building task matrices...")
    syco_matrices  = build_task_matrix(VECTOR_DIRS, LAYERS)
    seval_matrices = build_task_matrix(VECTOR_DIRS_SEMEVAL, LAYERS)

    syco_bases  = {lang: extract_subspace(M, k=K) for lang, M in syco_matrices.items()}
    seval_bases = {lang: extract_subspace(M, k=K) for lang, M in seval_matrices.items()}

    # Cross-task subspace overlap (same language)
    print("  Computing cross-task subspace overlap...")
    cross_task_overlap = {}
    for lang in LANGUAGES:
        if lang in syco_bases and lang in seval_bases:
            key = f"syco_vs_semeval_{lang}"
            cross_task_overlap[key] = subspace_overlap(syco_bases[lang], seval_bases[lang])
            info = cross_task_overlap[key]
            print(f"    {key}: mean={info['mean_overlap']:.4f}, "
                  f"max={info['max_overlap']:.4f}, "
                  f"angles={[round(a,1) for a in info['principal_angles_deg']]}")
    save_json(cross_task_overlap, OUTPUT_DIR / "subspace_overlap_cross_task.json")
    plot_subspace_overlap_bar(
        cross_task_overlap,
        title="Cross-Task Subspace Overlap (Sycophancy vs Syllogism)",
        save_path=OUTPUT_DIR / "subspace_overlap_cross_task.png")

    # Cross-lingual subspace overlap (same task)
    print("  Computing cross-lingual subspace overlap...")
    cross_lingual_overlap = {}
    lang_pairs_cl = [("en", "ko"), ("en", "es"), ("ko", "es")]
    for task, bases in [("syco", syco_bases), ("semeval", seval_bases)]:
        for l1, l2 in lang_pairs_cl:
            if l1 in bases and l2 in bases:
                key = f"{task}_{l1}_vs_{l2}"
                cross_lingual_overlap[key] = subspace_overlap(bases[l1], bases[l2])
                info = cross_lingual_overlap[key]
                print(f"    {key}: mean={info['mean_overlap']:.4f}, "
                      f"max={info['max_overlap']:.4f}")
    save_json(cross_lingual_overlap, OUTPUT_DIR / "subspace_overlap_cross_lingual.json")
    plot_subspace_overlap_bar(
        cross_lingual_overlap,
        title="Cross-Lingual Subspace Overlap",
        save_path=OUTPUT_DIR / "subspace_overlap_cross_lingual.png")

    # Projection ratio: syco vector onto semeval subspace per layer
    print("  Computing projection ratios...")
    proj_results = {}
    for lang in LANGUAGES:
        if lang not in seval_bases or lang not in vectors:
            continue
        df_proj = projection_ratio_per_layer(vectors, seval_bases[lang], lang, LAYERS)
        df_proj.to_csv(OUTPUT_DIR / f"projection_syco_onto_semeval_{lang}.csv", index=False)
        proj_results[lang] = df_proj
        print(f"    {lang}: mean proj_ratio={df_proj['proj_ratio'].mean():.4f}")

    # Projection vs performance correlation
    print("  Computing projection vs performance correlation...")
    perf_en_semeval_m1p5 = {
        12: -2.8, 13: -1.0, 14: -6.8, 15: -14.4, 16: -9.7,
        17: -7.8, 18: 7.0,  19: -6.8, 20: 12.2,  21: 2.0,
        22: 9.5,  23: 1.1,  24: 4.1,  25: 7.7,   26: -0.9
    }
    perf_ko_semeval_m1p5 = {
        12: -2.6, 13: -0.9, 14: 0.6,  15: -1.0,  16: -0.4,
        17: 4.1,  18: 1.8,  19: 1.4,  20: 3.9,   21: 1.7,
        22: 4.1,  23: 0.1,  24: 1.6,  25: 2.7,   26: 5.4
    }
    perf_es_semeval_m1p5 = {
        12: -2.1, 13: 2.2,  14: 4.4,  15: -1.5, 16: -4.4,
        17: -2.8, 18: 1.2,  19: -4.7, 20: 1.5,  21: 3.6,
        22: 0.4,  23: 0.5,  24: 8.3,  25: 5.1,  26: 0.3
    }
    for lang, perf_deltas in [("en", perf_en_semeval_m1p5),
                               ("ko", perf_ko_semeval_m1p5),
                               ("es", perf_es_semeval_m1p5)]:
        if lang not in proj_results:
            continue
        plot_projection_vs_performance(
            proj_results[lang], perf_deltas,
            title=f"Syco Vector Projection onto Semeval Subspace vs Syllogism Accuracy ({lang.upper()})",
            save_path=OUTPUT_DIR / f"projection_vs_performance_{lang}.png",
            lang=lang)

    # Final summary
    summary = {
        "languages": LANGUAGES,
        "layers": LAYERS,
        "vector_dim": int(unique_dims[0]),
        "output_dir": str(OUTPUT_DIR),
        "key_findings": findings,
        "cross_task_findings": findings_cross,
    }
    save_json(summary, OUTPUT_DIR / "analysis_summary.json")

    print(f"\nDone. All outputs saved to:\n{OUTPUT_DIR}")
