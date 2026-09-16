#!/usr/bin/env python3
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from parse_utils import parse_prediction as parse

def analyze(results_dir, prefix, source_path, lang):
    with open(source_path) as f:
        source = json.load(f)
    total = len(source)

    print(f"\n===== Coverage Analysis: {lang.upper()} =====")
    print(f"{'Layer':>6} | {'Parsed':>6} | {'Coverage':>8} | {'Delta':>7}")
    print("-" * 40)

    for layer in range(12, 27):
        result_file = Path(results_dir) / f"{prefix}_layer{layer}" / f"{prefix}_layer{layer}_results.json"
        if not result_file.exists():
            print(f"{layer:6d} | {'N/A':>6} | {'N/A':>8} | {'N/A':>7}")
            continue
        with open(result_file) as f:
            results = json.load(f)

        orig_correct = steer_correct = orig_parsed = steer_parsed = 0
        for i, item in enumerate(results):
            gold = source[i].get('matching', '')
            gold_label = '(A)' if '(A)' in gold else '(B)'

            orig_raw = item.get('orig_pred', '')
            orig_raw = orig_raw[0] if isinstance(orig_raw, list) else orig_raw
            orig_pred = parse(orig_raw)
            if orig_pred:
                orig_parsed += 1
                if orig_pred == gold_label: orig_correct += 1

            steer_raw = item.get('pred', '')
            steer_raw = steer_raw[0] if isinstance(steer_raw, list) else steer_raw
            steer_pred = parse(steer_raw)
            if steer_pred:
                steer_parsed += 1
                if steer_pred == gold_label: steer_correct += 1

        coverage = steer_parsed / total * 100
        orig_acc = orig_correct / orig_parsed if orig_parsed > 0 else 0
        steer_acc = steer_correct / steer_parsed if steer_parsed > 0 else 0
        delta = steer_acc - orig_acc
        print(f"{layer:6d} | {steer_parsed:6d} | {coverage:7.1f}% | {delta:+7.1%}")

if __name__ == "__main__":
    analyze("results", "semeval_en", "data/semeval/semeval_en_train.json", "en")
    analyze("results", "semeval_ko", "data/semeval/semeval_ko_train.json", "ko")
    analyze("results", "semeval_es", "data/semeval/semeval_es_train.json", "es")
