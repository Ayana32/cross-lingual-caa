"""
L2 normalize steering vectors per layer, following Rimsky et al.
Saves normalized vectors to a new directory: caa_vector_multiple_choice_normalized
"""
import torch
import numpy as np
from pathlib import Path

LAYERS = list(range(12, 27))

VECTOR_DIRS = {
    "en": Path("vectors/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice"),
    "ko": Path("vectors/Qwen2.5-7B-Instruct-ko/sycophancy_ko/caa_vector_multiple_choice"),
    "es": Path("vectors/Qwen2.5-7B-Instruct-es/sycophancy_es/caa_vector_multiple_choice"),
    "kk": Path("/users/acp25mk/EasyEdit/vectors/Qwen2.5-7B-Instruct-kk/sycophancy_kk/caa_vector_multiple_choice"),
}

for lang, src_dir in VECTOR_DIRS.items():
    out_dir = src_dir.parent / "caa_vector_multiple_choice_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for layer in LAYERS:
        src_path = src_dir / f"layer_{layer}.pt"
        if not src_path.exists():
            print(f"Missing: {src_path}")
            continue
        
        vec = torch.load(src_path, map_location="cpu", weights_only=True)
        norm = vec.norm().item()
        vec_normalized = vec / norm
        
        torch.save(vec_normalized, out_dir / f"layer_{layer}.pt")
        print(f"{lang} L{layer}: norm {norm:.4f} -> 1.0000")
    
    print(f"{lang} normalized vectors saved to {out_dir}\n")

print("Done!")
