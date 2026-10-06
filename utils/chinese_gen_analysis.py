#!/usr/bin/env python3
"""
Measures Chinese character generation rate across layers and multipliers.
Uses regex [\u4e00-\u9fff] to detect Chinese characters in model output.
"""
import json
import re
import argparse
from pathlib import Path


def has_chinese(text):
    """Return True if text contains any Chinese characters."""
    if isinstance(text, list):
        text = text[0] if text else ''
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def analyze_chinese_rate(results_dir, prefix):
    """Compute Chinese gen rate for a single result folder."""
    result_file = Path(results_dir) / f"{prefix}_results.json"
    if not result_file.exists():
        return None
    data = json.load(open(result_file))
    total = len(data)
    chinese = sum(1 for item in data if has_chinese(item.get('pred', '')))
    baseline = sum(1 for item in data if has_chinese(item.get('orig_pred', '')))
    return {
        "total": total,
        "chinese": chinese,
        "chinese_rate": chinese / total * 100,
        "baseline_chinese": baseline,
        "baseline_rate": baseline / total * 100,
    }


def compare_multipliers(base_results_dir, lang, task, layers=range(12, 27)):
    """Compare Chinese gen rate across multipliers per layer."""
    multipliers = {
        "m1.5":  f"{task}_{lang}_layer",
        "m1.0":  f"{task}_{lang}_m1.0_layer",
        "m-1.5": f"{task}_{lang}_m-1.5_layer",
    }

    print(f"\n=== Chinese Gen Rate: {task.upper()} {lang.upper()} ===")
    print(f"{'Layer':>6} | {'baseline':>9} | {'m-1.5':>8} | {'m1.0':>8} | {'m1.5':>8}")
    print("-" * 52)

    for layer in layers:
        row = [f"{layer:6d}"]
        baseline_rate = None
        for label, prefix_base in multipliers.items():
            folder = Path(base_results_dir) / f"{prefix_base}{layer}"
            prefix = f"{prefix_base}{layer}"
            result = analyze_chinese_rate(folder, prefix)
            if result is None:
                row.append(f"{'N/A':>8}")
            else:
                if baseline_rate is None:
                    baseline_rate = result["baseline_rate"]
                row.append(f"{result['chinese_rate']:>7.1f}%")
        baseline_str = f"{baseline_rate:>8.1f}%" if baseline_rate else f"{'N/A':>9}"
        print(f"{row[0]} | {baseline_str} | {row[3]} | {row[2]} | {row[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--task", default="semeval", choices=["semeval", "syco"])
    args = parser.parse_args()

    compare_multipliers(args.results_dir, args.lang, args.task)
