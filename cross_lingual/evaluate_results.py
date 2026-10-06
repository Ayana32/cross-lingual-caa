# -*- coding: utf-8 -*-
import json
import argparse
import wandb
from pathlib import Path

def parse_prediction(pred_text):
    """Extract (A) or (B) from model output. Invalid checked before valid."""
    if not pred_text:
        return None
    text = pred_text.strip().lower()

    # (B) / Invalid — checked first to avoid "invalid" matching "valid"
    if "(b)" in text or text == "b" or "invalid" in text:
        return "(B)"
    # (A) / Valid
    if "(a)" in text or text == "a" or "valid" in text:
        return "(A)"
    return None  # unparseable

def evaluate(results_path, log_wandb=True, run_name=None, source_path=None):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Load gold labels from source data if available
    gold_labels = {}
    if source_path and Path(source_path).exists():
        with open(source_path, "r", encoding="utf-8") as f:
            source = json.load(f)
        gold_labels = {i: item["matching"] for i, item in enumerate(source)}

    total = len(results)
    orig_stats = {"correct": 0, "A": 0, "B": 0, "unparseable": 0}
    steer_stats = {"correct": 0, "A": 0, "B": 0, "unparseable": 0}

    for idx, item in enumerate(results):
        gold = gold_labels.get(idx, None)
        if gold is None:
            continue
        gold_label = "(A)" if "(A)" in gold else "(B)"

        # Original output
        orig_pred_raw = item.get("orig_pred", "")
        orig_pred_raw = orig_pred_raw[0] if isinstance(orig_pred_raw, list) else orig_pred_raw
        orig_pred = parse_prediction(orig_pred_raw)
        if orig_pred is None:
            orig_stats["unparseable"] += 1
        else:
            orig_stats[orig_pred.strip("()")] += 1
            if orig_pred == gold_label:
                orig_stats["correct"] += 1

        # Steered output
        steer_pred_raw = item.get("pred", "")
        steer_pred_raw = steer_pred_raw[0] if isinstance(steer_pred_raw, list) else steer_pred_raw
        steer_pred = parse_prediction(steer_pred_raw)
        if steer_pred is None:
            steer_stats["unparseable"] += 1
        else:
            steer_stats[steer_pred.strip("()")] += 1
            if steer_pred == gold_label:
                steer_stats["correct"] += 1

    parseable_orig  = total - orig_stats["unparseable"]
    parseable_steer = total - steer_stats["unparseable"]

    orig_acc  = orig_stats["correct"]  / parseable_orig  if parseable_orig  > 0 else 0
    steer_acc = steer_stats["correct"] / parseable_steer if parseable_steer > 0 else 0

    metrics = {
        "total": total,
        # Original
        "orig/accuracy":          round(orig_acc, 4),
        "orig/valid_rate":        round(orig_stats["A"]  / parseable_orig,  4) if parseable_orig  > 0 else 0,
        "orig/invalid_rate":      round(orig_stats["B"]  / parseable_orig,  4) if parseable_orig  > 0 else 0,
        "orig/unparseable":       orig_stats["unparseable"],
        # Steered
        "steer/accuracy":         round(steer_acc, 4),
        "steer/valid_rate":       round(steer_stats["A"] / parseable_steer, 4) if parseable_steer > 0 else 0,
        "steer/invalid_rate":     round(steer_stats["B"] / parseable_steer, 4) if parseable_steer > 0 else 0,
        "steer/unparseable":      steer_stats["unparseable"],
        # Delta
        "delta/accuracy":         round(steer_acc - orig_acc, 4),
    }

    print(f"\n===== Evaluation: {Path(results_path).stem} =====")
    print(f"Total items      : {total}")
    print(f"\n[Original]")
    print(f"  Accuracy       : {metrics['orig/accuracy']:.1%}")
    print(f"  Valid rate     : {metrics['orig/valid_rate']:.1%}")
    print(f"  Invalid rate   : {metrics['orig/invalid_rate']:.1%}")
    print(f"  Unparseable    : {orig_stats['unparseable']}")
    print(f"\n[Steered]")
    print(f"  Accuracy       : {metrics['steer/accuracy']:.1%}")
    print(f"  Valid rate     : {metrics['steer/valid_rate']:.1%}")
    print(f"  Invalid rate   : {metrics['steer/invalid_rate']:.1%}")
    print(f"  Unparseable    : {steer_stats['unparseable']}")
    print(f"\n[Delta]")
    print(f"  Accuracy delta : {metrics['delta/accuracy']:+.1%}")

    if log_wandb:
        run_name = run_name or Path(results_path).stem
        wandb.init(project="cross-lingual-caa", name=run_name, reinit=True)
        wandb.log(metrics)
        wandb.finish()
        print(f"\nLogged to W&B: {run_name}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="Path to results JSON file")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--source", default=None, help="Path to original source JSON with gold labels")
    args = parser.parse_args()

    evaluate(args.results_path, log_wandb=not args.no_wandb, run_name=args.run_name, source_path=args.source)
