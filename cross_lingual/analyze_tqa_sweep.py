# -*- coding: utf-8 -*-
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from parse_utils import parse_prediction


def evaluate_single_layer(results, source, task="truthfulqa"):
    gold_labels = {i: item["matching"] for i, item in enumerate(source)}
    total = len(results)
    orig_stats  = {"correct": 0, "A": 0, "B": 0, "unparseable": 0}
    steer_stats = {"correct": 0, "A": 0, "B": 0, "unparseable": 0}

    for idx, item in enumerate(results):
        gold = gold_labels.get(idx)
        if gold is None:
            continue
        gold_label = "(A)" if "(A)" in gold else "(B)"

        orig_raw = item.get("orig_pred", "")
        orig_raw = orig_raw[0] if isinstance(orig_raw, list) else orig_raw
        orig_pred = parse_prediction(orig_raw)

        steer_raw = item.get("pred", "")
        steer_raw = steer_raw[0] if isinstance(steer_raw, list) else steer_raw
        steer_pred = parse_prediction(steer_raw)

        if orig_pred is None or steer_pred is None:
            orig_stats["unparseable"] += int(orig_pred is None)
            steer_stats["unparseable"] += int(steer_pred is None)
            continue

        orig_stats[orig_pred.strip("()")] += 1
        if orig_pred == gold_label:
            orig_stats["correct"] += 1

        steer_stats[steer_pred.strip("()")] += 1
        if steer_pred == gold_label:
            steer_stats["correct"] += 1

    paired_total = sum(orig_stats[k] for k in ["A", "B"])
    orig_rate  = orig_stats["correct"] / paired_total if paired_total > 0 else 0
    steer_rate = steer_stats["correct"] / paired_total if paired_total > 0 else 0

    return {
        "total": total,
        "paired": paired_total,
        "orig_accuracy":  round(orig_rate, 4),
        "steer_accuracy": round(steer_rate, 4),
        "delta_accuracy": round(steer_rate - orig_rate, 4),
        "orig_unparseable":  orig_stats["unparseable"],
        "steer_unparseable": steer_stats["unparseable"],
    }


def analyze_tqa_sweep(results_dir, source_path, lang="en", run_name=None):
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    source = json.load(open(source_path, encoding="utf-8"))
    all_metrics = {}

    print(f"\n===== TQA Sweep Analysis: {lang.upper()} =====")
    print(f"{'File':<40} {'Orig Acc':>12} {'Steer Acc':>12} {'Delta':>8} {'Unparse':>8}")
    print("-" * 86)

    for rf in result_files:
        results = json.load(open(rf, encoding="utf-8"))
        name = rf.stem
        m = evaluate_single_layer(results, source, task="truthfulqa")
        all_metrics[name] = m
        print(f"{name:<40} {m['orig_accuracy']:>11.1%} {m['steer_accuracy']:>11.1%} "
              f"{m['delta_accuracy']:>+7.1%} {m['steer_unparseable']:>8}")

    best_delta = max(all_metrics.items(), key=lambda x: x[1]["delta_accuracy"])
    best_steer = max(all_metrics.items(), key=lambda x: x[1]["steer_accuracy"])
    print(f"\nBest delta    : {best_delta[0]} -> {best_delta[1]['delta_accuracy']:+.1%}")
    print(f"Best accuracy : {best_steer[0]} -> {best_steer[1]['steer_accuracy']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--source", required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    analyze_tqa_sweep(args.results_dir, args.source, lang=args.lang)
