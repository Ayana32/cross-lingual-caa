#!/usr/bin/env python3
"""
Geometry vs behaviour (TruthfulQA):
  x = EN<->target native steering-vector cosine (EN-selected layer)
  y = TQA behavioural delta when EN-derived vector is applied to the target (pp)
Qwen and OLMo3 only. Gemma excluded (TQA ceiling).
Key visual: a shared-similarity band showing same x, opposite y.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {
    "Qwen3.5 (L18)": {
        "color": "#C0392B",
        "points": {"IT": (0.984, 0.0), "ZH": (0.980, 1.0), "KO": (0.969, -1.0), "KK": (0.945, -1.0)},
    },
    "OLMo3 (L14)": {
        "color": "#2E86AB",
        "points": {"IT": (0.948, 9.0), "ZH": (0.960, 9.0), "KO": (0.916, 8.0), "KK": (0.758, 12.0)},
    },
}

fig, ax = plt.subplots(figsize=(9, 6))

# Shared-similarity band: same cosine range, opposite behaviour
ax.axvspan(0.94, 0.985, color="#999999", alpha=0.13, zorder=0)
ax.text(0.9625, 15.2, "Same similarity,\nopposite behaviour",
        fontsize=10.5, ha="center", va="top", color="#444444", fontweight="bold")


ax.axhline(0, color="#888888", lw=1, ls="--", zorder=1)

for label, d in DATA.items():
    xs, ys, names = [], [], []
    for lang, (x, y) in d["points"].items():
        xs.append(x); ys.append(y); names.append(lang)
    ax.scatter(xs, ys, s=180, color=d["color"], edgecolor="white",
               linewidth=1.5, label=label, zorder=3)
    for x, y, lang in zip(xs, ys, names):
        ax.annotate(lang, (x, y), textcoords="offset points", xytext=(9, 5),
                    fontsize=11, color=d["color"], fontweight="bold")

# Qwen cluster callout (point at IT/ZH cluster)
ax.annotate("high similarity,\ntransfer ≈ 0",
            xy=(0.982, 0.5), xytext=(0.885, 3.2), fontsize=10,
            color="#C0392B", ha="center",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

# OLMo3 KK callout (text above-right, arrow up to point)
ax.annotate("lowest similarity,\nlargest observed Δ",
            xy=(0.758, 12.0), xytext=(0.79, 14.2), fontsize=10,
            color="#2E86AB", ha="left",
            arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.2))

ax.set_xlabel("EN–target steering-vector cosine similarity", fontsize=12)
ax.set_ylabel("TQA Δ under EN-derived steering (pp)", fontsize=12)
ax.set_title("Cross-lingual steering-vector similarity vs. behavioural transfer",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.70, 1.0)
ax.set_ylim(-6, 16.5)
ax.legend(loc="center left", fontsize=11, frameon=True)

ax.text(0.72, -5.5, "Gemma excluded due to TQA ceiling effects (baseline 89–91%).",
        fontsize=9, style="italic", color="#555555")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("figures/geometry_vs_behaviour_tqa.png", dpi=200, bbox_inches="tight")
print("saved figures/geometry_vs_behaviour_tqa.png")
