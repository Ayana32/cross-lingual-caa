#!/usr/bin/env python3
"""
score_controls.py  --  score all control runs with YOUR evaluate_results.py and
emit results_long.csv for analyze_random_baseline.py.

It does NOT reimplement scoring: it imports evaluate() from scripts/evaluate_results.py
so the control numbers are produced by the exact same logic (and parse rules) that
gave the -42.8pp headline. That identical-scoring property is what makes the
control comparison fair.

Metric mapping (sycophancy task):
  evaluate() returns accuracy vs gold `matching`, and matching = the sycophantic
  answer, so  steer/accuracy  == steered sycophancy rate, and
              orig/accuracy   == baseline (unsteered) sycophancy rate.
We emit sycophancy rate in %, plus the unparseable count (an over-steering /
output-breakdown signal worth watching).

USAGE
  python score_controls.py \
      --control_root results/control \
      --real_dir   strong=results/syco_en_layer17 \
      --source     data/sycophancy_en_train.json \
      --language en --task sycophancy --layer 17 --eval generation \
      --out results_long.csv

  --real_dir takes one or more  regime=path  pairs (the real-vector result dir for
  that regime). The baseline row is taken from that real run's orig/accuracy.

Then:
  python analyze_random_baseline.py --csv results_long.csv --out_dir analysis
"""

import argparse
import csv
import glob
import os
import sys


def find_results_json(d):
    """Return the single *_results.json inside a result dir."""
    hits = sorted(glob.glob(os.path.join(d, "*results.json")))
    if not hits:
        raise FileNotFoundError(f"no *_results.json in {d}")
    return hits[0]


def load_evaluator(scripts_dir):
    sys.path.insert(0, scripts_dir)
    from evaluate_results import evaluate  # noqa: their script
    return evaluate


# ---- syllogism (validity-based) scoring -------------------------------------
# semeval gold = correct logical answer: validity True -> (A) Valid, False -> (B).
# This differs from sycophancy, where gold = `matching` (the sycophantic answer).
def _parse_ab(text):
    if not text:
        return None
    t = text.strip().lower()
    if "(b)" in t or t == "b" or "invalid" in t:
        return "(B)"
    if "(a)" in t or t == "a" or "valid" in t:
        return "(A)"
    return None


def evaluate_syllogism(results_path, source_path, gold_field="validity"):
    """Return metrics dict matching evaluate()'s keys, but scored on logical
    correctness (gold from `validity`) instead of `matching`."""
    import json
    results = json.load(open(results_path, encoding="utf-8"))
    source = json.load(open(source_path, encoding="utf-8"))
    gold = ["(A)" if bool(s[gold_field]) else "(B)" for s in source]

    def acc(key):
        correct = unparse = 0
        n = min(len(results), len(gold))
        for i in range(n):
            raw = results[i].get(key, "")
            raw = raw[0] if isinstance(raw, list) else raw
            p = _parse_ab(raw)
            if p is None:
                unparse += 1
            elif p == gold[i]:
                correct += 1
        parseable = n - unparse
        return (correct / parseable if parseable else 0.0), unparse, n

    o_acc, o_un, n = acc("orig_pred")
    s_acc, s_un, _ = acc("pred")
    return {
        "total": n,
        "orig/accuracy": round(o_acc, 4), "orig/unparseable": o_un,
        "steer/accuracy": round(s_acc, 4), "steer/unparseable": s_un,
        "delta/accuracy": round(s_acc - o_acc, 4),
    }


def regime_of(name):
    # control subdir names look like syco_en_L17_strong / syco_en_L17_corrected
    if name.endswith("_strong"):
        return "strong"
    if name.endswith("_corrected"):
        return "corrected"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_root", required=True)
    ap.add_argument("--real_dir", nargs="+", required=True,
                    help="one or more regime=path pairs, e.g. strong=results/syco_en_layer17")
    ap.add_argument("--source", required=True, help="gold JSON with `matching` labels")
    ap.add_argument("--scripts_dir", default="scripts")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="sycophancy", choices=["sycophancy", "syllogism"])
    ap.add_argument("--layer", default="17")
    ap.add_argument("--eval", default="generation")
    ap.add_argument("--gold_field", default="validity",
                    help="source field for syllogism gold (default: validity)")
    ap.add_argument("--out", default="results_long.csv")
    args = ap.parse_args()

    real_dirs = dict(pair.split("=", 1) for pair in args.real_dir)

    if args.task == "syllogism":
        def score(path):
            return evaluate_syllogism(path, args.source, gold_field=args.gold_field)
    else:
        evaluate = load_evaluator(args.scripts_dir)
        def score(path):
            return evaluate(path, log_wandb=False, source_path=args.source)

    rows = []

    def base(regime, condition, method, seed, metric, unparseable):
        return {
            "regime": regime, "language": args.language, "task": args.task,
            "layer": args.layer, "eval": args.eval, "condition": condition,
            "method": method, "seed": seed, "metric": round(metric, 2),
            "unparseable": unparseable,
        }

    # --- real + baseline (one per regime) ---
    for regime, rdir in real_dirs.items():
        m = score(find_results_json(rdir))
        rows.append(base(regime, "real", "na", "na",
                         m["steer/accuracy"] * 100, m["steer/unparseable"]))
        rows.append(base(regime, "baseline", "na", "na",
                         m["orig/accuracy"] * 100, m["orig/unparseable"]))
        print(f"[real:{regime}] sycophancy rate: baseline {m['orig/accuracy']*100:.1f}% "
              f"-> steered {m['steer/accuracy']*100:.1f}% "
              f"(delta {m['delta/accuracy']*100:+.1f}pp)")

    # --- controls ---
    for name_dir in sorted(glob.glob(os.path.join(args.control_root, "*"))):
        if not os.path.isdir(name_dir):
            continue
        regime = regime_of(os.path.basename(name_dir))
        for run_dir in sorted(glob.glob(os.path.join(name_dir, "*_seed*"))):
            tag = os.path.basename(run_dir)          # gaussian_seed0
            method, seed = tag.rsplit("_seed", 1)
            m = score(find_results_json(run_dir))
            rows.append(base(regime, "random", method, int(seed),
                             m["steer/accuracy"] * 100, m["steer/unparseable"]))

    fields = ["regime", "language", "task", "layer", "eval", "condition",
              "method", "seed", "metric", "unparseable"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {args.out}")
    print("next: python analyze_random_baseline.py --csv "
          f"{args.out} --out_dir analysis")


if __name__ == "__main__":
    main()
