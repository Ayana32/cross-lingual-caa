"""Recompute TruthfulQA delta-TR and baselines from raw outputs, with bootstrap CI.

Run from the repo root on Stanage:  python check_tqa.py
Check 0: fingerprint of each eval.json (a changed eval file would shift baselines).
Check 1: OLMo3, every language x layer. Answers KO L14 (+4.0 vs +8.0) and shows
         whether the unsteered baseline is the same across layer runs
         (8/17 report: KO 64.0, KK 57.0; tqa_summary.txt: KO 56.0, KK 53.0).
Check 2: every Qwen3.5 L18 result file (pre-fix vs post-fix run).
"""
import glob
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, "cross_lingual")
from parse_utils import parse_prediction

LANGS = ["en", "it", "zh", "ko", "kk"]


def parse(text, lang):
    try:
        return parse_prediction(text, lang)
    except TypeError:
        return parse_prediction(text)


def norm(x):
    if x is None:
        return None
    x = str(x).strip().strip("()").strip().upper()
    return x if x in ("A", "B") else None


def mtime(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def evaluate(path, lang, n_boot=1000, seed=42):
    src = json.load(open(f"data/truthfulqa/{lang}/eval.json"))
    res = json.load(open(path))
    if len(src) != len(res):
        raise ValueError(f"length mismatch: eval {len(src)} vs results {len(res)}")
    pairs, base_all, unparse = [], [], 0
    for s, r in zip(src, res):
        gold = norm(s["matching"])
        b = norm(parse(r["orig_pred"], lang))
        st = norm(parse(r["pred"], lang))
        if b is not None:
            base_all.append(b == gold)
        if b is None or st is None:
            unparse += 1
            continue
        pairs.append((b == gold, st == gold))
    n = len(pairs)
    base = 100 * sum(p[0] for p in pairs) / n
    steer = 100 * sum(p[1] for p in pairs) / n
    base_unpaired = 100 * sum(base_all) / len(base_all)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        smp = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(100 * (sum(p[1] for p in smp) - sum(p[0] for p in smp)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return n, unparse, base, base_unpaired, steer, steer - base, lo, hi


def row(tag, path, lang):
    try:
        n, u, b, bu, s, d, lo, hi = evaluate(path, lang)
        sig = "yes" if lo > 0 or hi < 0 else "no"
        print(f"{tag:<26} N={n:<4} unp={u:<3} base={b:5.1f} (unpaired {bu:5.1f}) "
              f"steer={s:5.1f} d={d:+5.1f} CI=[{lo:+5.1f},{hi:+5.1f}] excl0={sig}  {mtime(path)}")
    except Exception as e:
        print(f"{tag:<26} ERROR {e}  {mtime(path)}")


print("=== Check 0: eval.json fingerprints ===")
for lang in LANGS:
    p = f"data/truthfulqa/{lang}/eval.json"
    if os.path.exists(p):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
        print(f"{lang.upper()}  md5={h}  modified {mtime(p)}")

print("\n=== Check 1: OLMo3, all languages x layers (m1.5) ===")
for lang in LANGS:
    for layer in range(10, 24):
        p = f"results/tqa_olmo3_{lang}_m1.5_layer{layer}/tqa_{lang}_results.json"
        if os.path.exists(p):
            flag = "  <--" if (lang == "ko" and layer == 14) else ""
            row(f"OLMo3 {lang.upper()} L{layer}{flag}", p, lang)
    print()

print("=== Check 2: every Qwen3.5 L18 result file ===")
for lang in LANGS:
    hits = sorted(glob.glob(f"results/*qwen35*{lang}*layer18*/**/*.json", recursive=True))
    if not hits:
        print(f"Qwen3.5 {lang.upper()} L18: no files found (check the glob)")
    for p in hits:
        row(f"Qwen3.5 {lang.upper()} L18", p, lang)
        print(f"    {p}")
