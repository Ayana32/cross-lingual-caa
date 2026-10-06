#!/usr/bin/env python3
"""
Compute EN->target cross-lingual cosine similarity for TQA steering vectors.

Matches the sycophancy geometry definition: cosine between the EN-derived vector
and each target language's native vector, at the SAME layer used to read behaviour
(the English-selected layer). Reports that layer; also prints the full sweep as a
backup so the reported value is visibly not cherry-picked.
"""
import os
import numpy as np
import torch

BASE = "vectors/{model_dir}/truthfulqa/caa_vector/truthfulqa_{lang}/caa_vector_multiple_choice/layer_{layer}.pt"

MODELS = {
    "Qwen3.5": {"model_dir": "Qwen3.5-9B", "en_layer": 18, "layers": range(10, 24)},
    "Gemma4":  {"model_dir": "Gemma4-12B", "en_layer": 23, "layers": range(20, 34)},
    "OLMo3":   {"model_dir": "Olmo-3-7B-Instruct", "en_layer": 14, "layers": range(10, 24)},
}
TARGETS = ["it", "zh", "ko", "kk"]


def load_unit(model_dir, lang, layer):
    path = BASE.format(model_dir=model_dir, lang=lang, layer=layer)
    if not os.path.exists(path):
        return None
    v = torch.load(path, weights_only=True).float().numpy().ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cosine(a, b):
    return float(np.dot(a, b))


for name, cfg in MODELS.items():
    md, en_layer, layers = cfg["model_dir"], cfg["en_layer"], cfg["layers"]
    print(f"\n===== {name}  (EN-selected layer L{en_layer}) =====")

    en_vec = load_unit(md, "en", en_layer)
    if en_vec is None:
        print(f"  EN vector missing at L{en_layer}: {BASE.format(model_dir=md, lang='en', layer=en_layer)}")
        continue
    print(f"  Reported (matched-layer) EN<->target cosine at L{en_layer}:")
    for lang in TARGETS:
        tv = load_unit(md, lang, en_layer)
        if tv is None:
            print(f"    EN-{lang.upper()}: native vector missing at L{en_layer}")
        else:
            print(f"    EN-{lang.upper()}: {cosine(en_vec, tv):.3f}")

    print(f"  Backup sweep (EN<->target cosine by layer):")
    print("    layer " + " ".join(f"{l.upper():>6}" for l in TARGETS))
    for L in layers:
        env = load_unit(md, "en", L)
        if env is None:
            continue
        cells = []
        for lang in TARGETS:
            tv = load_unit(md, lang, L)
            cells.append(f"{cosine(env, tv):+.3f}" if tv is not None else "  -   ")
        marker = " <-- EN-selected" if L == en_layer else ""
        print(f"    L{L:<4} " + " ".join(f"{c:>6}" for c in cells) + marker)
