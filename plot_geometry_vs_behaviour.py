#!/usr/bin/env python3
"""
Figure: EN<->target TruthfulQA steering-vector cosine (EN-selected layer)
vs. TruthfulQA delta-TR under EN-derived steering, with 95% bootstrap CIs.

Cosines: results/tqa_cosine_matched.txt (compute_tqa_cosine.py, verified 5 Oct 2026).
Delta-TR and CIs: raw-output recompute (check_tqa.py; paired parseable items,
1,000 bootstrap resamples, seed 42), identical to Dissertation Table 8.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# lang: (cosine, delta, ci_low, ci_high)
DATA = {
    "Qwen3.5 (L18)": {
        "color": "#c0392b",
        "points": {
            "IT": (0.984, 0.0, -5.1, 6.1),
            "ZH": (0.980, 1.0, -4.0, 6.1),
            "KO": (0.969, -1.0, -5.0, 3.0),
            "KK": (0.945, -1.0, -6.0, 4.0),
        },
    },
    "OLMo3 (L14)": {
        "color": "#2e86ab",
        "points": {
            "IT": (0.948, 9.0, 1.0, 17.0),
            "ZH": (0.960, 9.0, 4.0, 15.0),
            "KO": (0.916, 4.0, -3.0, 11.0),
            "KK": (0.758, 12.0, 1.0, 22.0),
        },
    },
}
LABEL_OFFSET = {  # Qwen labels sit beyond the CI caps (see LABEL_AT_CAP)
    ("Qwen3.5 (L18)", "IT"): (0, -14), ("Qwen3.5 (L18)", "ZH"): (0, 5),
    ("Qwen3.5 (L18)", "KO"): (0, -14), ("Qwen3.5 (L18)", "KK"): (0, -14),
    ("OLMo3 (L14)", "IT"): (-24, 8), ("OLMo3 (L14)", "ZH"): (8, 8),
    ("OLMo3 (L14)", "KO"): (8, 6), ("OLMo3 (L14)", "KK"): (8, 6),
}

fig, ax = plt.subplots(figsize=(9, 6))
ax.axvspan(0.94, 0.987, color="#999999", alpha=0.13, zorder=0)
ax.text(0.9635, 23.2, "Similar alignment,\ndifferent behaviour", ha="center",
        va="top", fontsize=10.5, fontweight="bold", color="#444444")
ax.axhline(0, color="#888888", linestyle="--", linewidth=1, zorder=1)

for model, cfg in DATA.items():
    first = True
    for lang, (x, y, lo, hi) in cfg["points"].items():
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=cfg["color"],
                    ecolor=cfg["color"], elinewidth=1.4, capsize=4, markersize=11,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3,
                    label=model if first else None, alpha=0.95)
        first = False
        dx, dy = LABEL_OFFSET[(model, lang)]
        if model.startswith("Qwen"):
            anchor, ha = (x, hi if dy > 0 else lo), "center"
        else:
            anchor, ha = (x, y), "left"
        ax.annotate(lang, anchor, xytext=(dx, dy), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=cfg["color"], ha=ha)

ax.annotate("lowest similarity,\nlargest observed Δ", xy=(0.760, 12.0),
            xytext=(0.785, 18.5), fontsize=10, color="#2e86ab",
            arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.4))
ax.annotate("high similarity,\nΔ ≈ 0", xy=(0.941, -1.2), xytext=(0.895, -6.8),
            fontsize=10, color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))

ax.set_xlim(0.70, 1.00)
ax.set_ylim(-10, 24)
ax.set_xlabel("EN–target steering-vector cosine similarity", fontsize=12)
ax.set_ylabel("TruthfulQA ΔTR under EN-derived steering (pp)", fontsize=12)
ax.set_title("Cross-lingual steering-vector similarity vs. behavioural effect",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, zorder=0)
ax.legend(loc="center left", fontsize=11, framealpha=0.95)
ax.text(0.705, -9.5, "Error bars: 95% bootstrap CIs (1,000 resamples). "
        "Gemma4 excluded (TruthfulQA baseline 89–91%).",
        fontsize=8.5, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("figures/geometry_vs_behaviour_tqa.png", dpi=200, bbox_inches="tight")
plt.savefig("figures/geometry_vs_behaviour_tqa.pdf", bbox_inches="tight")
print("saved figures/geometry_vs_behaviour_tqa.png and .pdf")
