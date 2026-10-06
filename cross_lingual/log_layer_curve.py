#!/usr/bin/env python3
import json
import argparse
import wandb
from pathlib import Path

def parse_prediction(pred_text):
    if not pred_text:
        return None
    text = pred_text.strip().lower()
    if "(b)" in text or text == "b" or "invalid" in text:
        return "(B)"
    if "(a)" in text or text == "a" or "valid" in text:
        return "(A)"
    return None

def evaluate(results, source):
    gold_labels = {i: item["matching"] for i, item in enumerate(source)}
    orig_correct = steer_correct = orig_unparse = steer_unparse = 0
    total = len(results)
    for idx, item in enumerate(results):
        gold = gold_labels.get(idx)
        if gold is None:
            continue
        gold_label = "(A)" if "(A)" in gold else "(B)"
        orig_raw = item.get("orig_pred", "")
        orig_raw = orig_raw[0] if isinstance(orig_raw, list) else orig_raw
        orig_pred = parse_prediction(orig_raw)
        if orig_pred is None:
            orig_unparse += 1
        elif orig_pred == gold_label:
            orig_correct += 1
        steer_raw = item.get("pred", "")
        steer_raw = steer_raw[0] if isinstance(steer_raw, list) else steer_raw
        steer_pred = parse_prediction(steer_raw)
        if steer_pred is None:
            steer_unparse += 1
        elif steer_pred == gold_label:
            steer_correct += 1
    parseable_orig = total - orig_unparse
    parseable_steer = total - steer_unparse
    orig_acc = orig_correct / parseable_orig if parseable_orig > 0 else 0
    steer_acc = steer_correct / parseable_steer if parseable_steer > 0 else 0
    return orig_acc, steer_acc, steer_acc - orig_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--task", default="syllogism")
    parser.add_argument("--prefix", default=None, help="folder prefix, e.g. semeval_en or syco_en")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 27)))
    args = parser.parse_args()

    prefix = args.prefix if args.prefix else f"semeval_{args.lang}"

    with open(args.source) as f:
        source = json.load(f)

    run = wandb.init(
        project="cross-lingual-caa",
        name=f"layer-curve-{args.lang}-{args.task}",
        config={"lang": args.lang, "task": args.task, "layers": args.layers}
    )

    for layer in args.layers:
        result_file = Path(args.results_dir) / f"{prefix}_layer{layer}" / f"{prefix}_layer{layer}_results.json"
        if not result_file.exists():
            print(f"Missing: {result_file}")
            continue
        with open(result_file) as f:
            results = json.load(f)
        orig_acc, steer_acc, delta = evaluate(results, source)
        wandb.log({
            f"{args.lang}/layer": layer,
            f"{args.lang}/orig_acc": orig_acc,
            f"{args.lang}/steer_acc": steer_acc,
            f"{args.lang}/delta": delta,
        }, step=layer)
        print(f"Layer {layer}: orig={orig_acc:.3f} steer={steer_acc:.3f} delta={delta:+.3f}")

    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()
