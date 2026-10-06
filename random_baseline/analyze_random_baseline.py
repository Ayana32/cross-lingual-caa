#!/usr/bin/env python3
"""
analyze_random_baseline.py
===========================
Aggregate steering results, build the RANDOM control distribution, and test
whether the REAL steering vector's effect lies OUTSIDE that distribution.

This is the statistical core of the control experiment: it answers
"is the real vector's effect distinguishable from a matched random direction?"
by reporting, per condition, the z-score and a two-sided empirical p-value of
the real effect against the random-seed distribution.

INPUT  (a CSV you fill from your EasyEdit eval outputs) -- columns:
  regime      : corrected | strong
  language    : en | ko | es | kk
  task        : sycophancy | syllogism
  layer       : e.g. 17
  eval        : generation | token_prob
  condition   : baseline | real | random
  method      : na (baseline/real) | gaussian | permute   (only for random rows)
  seed        : integer (na for baseline/real)
  metric      : the measured value
                - sycophancy task -> sycophancy rate in % (LOWER = better; a
                  negative delta vs baseline = successful sycophancy reduction)
                - syllogism task  -> accuracy in %        (HIGHER = better)

One CSV can hold every (regime x language x task x layer x eval) combination.

USAGE
  python analyze_random_baseline.py --csv results_long.csv --out_dir analysis
  -> prints a per-group report and writes analysis/<group>.png + summary.csv

DELTA CONVENTION (kept explicit so the sign is never ambiguous):
  delta = metric(condition) - metric(baseline)
  sycophancy: delta < 0  => sycophancy reduced  => intended effect
  syllogism : delta > 0  => accuracy improved    => intended effect
The script reports raw deltas; interpret the sign per task as above.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["metric"] = float(r["metric"])
            r["layer"] = str(r["layer"]).strip()
            rows.append(r)
    return rows


def group_key(r):
    return (r["regime"], r["language"], r["task"], r["layer"], r["eval"])


def empirical_p(real, random_vals):
    """Two-sided empirical p-value of `real` against the random distribution,
    with +1 smoothing so p is never exactly 0 with finite samples."""
    rv = np.asarray(random_vals, float)
    center = rv.mean()
    obs = abs(real - center)
    as_extreme = np.sum(np.abs(rv - center) >= obs)
    return (as_extreme + 1) / (len(rv) + 1)


def analyze(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    summary = []
    for key in sorted(groups):
        regime, lang, task, layer, ev = key
        g = groups[key]
        baseline = next((x["metric"] for x in g if x["condition"] == "baseline"), None)
        real = next((x["metric"] for x in g if x["condition"] == "real"), None)
        rand = [x for x in g if x["condition"] == "random"]
        if baseline is None or real is None or not rand:
            print(f"[skip] {key}: needs baseline + real + >=1 random row")
            continue

        title = f"{regime} | {lang} | {task} | L{layer} | {ev}"
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        print(f"  baseline       : {baseline:7.2f}")
        print(f"  real           : {real:7.2f}   (delta {real - baseline:+.2f})")

        # per-method stats + pooled
        by_method = defaultdict(list)
        for x in rand:
            by_method[x.get("method", "random")].append(x["metric"])

        pooled = [x["metric"] for x in rand]
        for method, vals in list(by_method.items()) + [("ALL", pooled)]:
            arr = np.asarray(vals, float)
            mu, sd = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
            z = (real - mu) / sd if sd > 0 else float("nan")
            p = empirical_p(real, arr)
            verdict = ""
            if sd > 0 and abs(z) >= 2:
                verdict = "  REAL outside random (>=2 std) -> directional effect"
            elif sd > 0:
                verdict = "  REAL within random -> effect NOT direction-specific"
            print(f"  random[{method:8}] n={len(arr)} mean={mu:7.2f} sd={sd:5.2f} "
                  f"(delta {mu - baseline:+.2f}) | z={z:+.2f} p={p:.3f}{verdict}")
            summary.append({
                "regime": regime, "language": lang, "task": task, "layer": layer,
                "eval": ev, "method": method, "baseline": baseline, "real": real,
                "real_delta": real - baseline, "random_mean": mu, "random_sd": sd,
                "random_mean_delta": mu - baseline, "z": z, "p": p, "n_random": len(arr),
            })

        _plot_group(title, baseline, real, by_method, task,
                    os.path.join(out_dir, f"{regime}_{lang}_{task}_L{layer}_{ev}.png"))

    # write summary
    if summary:
        keys = list(summary[0].keys())
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
        print(f"\n[done] summary -> {os.path.join(out_dir, 'summary.csv')}")


def _plot_group(title, baseline, real, by_method, task, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"gaussian": "#4C78A8", "permute": "#E45756", "random": "#72B7B2"}

    # baseline reference
    ax.axhline(baseline, color="#999999", ls="--", lw=1.2, label="baseline (no steer)")

    # random distributions as jittered strips + mean band
    xpos = {m: i for i, m in enumerate(sorted(by_method))}
    rng = np.random.default_rng(0)
    for m, vals in by_method.items():
        arr = np.asarray(vals, float)
        x = xpos[m] + 0.0
        jitter = (rng.random(len(arr)) - 0.5) * 0.25
        ax.scatter(np.full(len(arr), x) + jitter, arr, s=28,
                   color=colors.get(m, "#888"), alpha=0.7, label=f"random:{m}")
        mu, sd = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
        ax.errorbar(x, mu, yerr=sd, fmt="_", color=colors.get(m, "#888"),
                    ms=22, mew=2.5, capsize=6, lw=2)

    # real vector as a star spanning all method columns
    real_x = (len(by_method) - 1) / 2.0
    ax.scatter([real_x], [real], marker="*", s=320, color="#F58518",
               edgecolor="black", zorder=5, label="real vector")

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(list(xpos.keys()))
    ylab = "sycophancy rate (%)  [lower = better]" if task == "sycophancy" \
           else "accuracy (%)  [higher = better]"
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="analysis")
    args = ap.parse_args()
    analyze(load_rows(args.csv), args.out_dir)


if __name__ == "__main__":
    main()
