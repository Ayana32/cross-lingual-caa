#!/usr/bin/env python3
"""
make_random_baseline_vectors.py
================================
Generate norm-matched RANDOM and PERMUTED control vectors from a real CAA
steering vector, for use as control baselines in activation-steering experiments.

WHY (design rationale, for the supervisor):
  A control baseline must isolate DIRECTION from MAGNITUDE. We therefore build
  control vectors whose L2 norm exactly matches the real vector that the apply
  config loads, so the only thing that differs between "real" and "control" runs
  is the direction. This follows the standard validation practice in the steering
  literature (e.g. comparing a learned vector against Gaussian-noise / random-
  direction baselines, with the random direction scaled to the CAA norm).

  Two control families are produced:
    - gaussian : an isotropic random direction, rescaled to the real norm.
                 -> "would ANY direction of this magnitude do the same?"
    - permute  : the real vector's components randomly shuffled.
                 -> preserves the norm AND the component-value distribution,
                    destroying only the direction. A STRONGER control than
                    gaussian, because it rules out "the effect is just a
                    consequence of the vector's component statistics".

TWO MAGNITUDE REGIMES:
  Run this once per "real reference vector" the apply config actually loads.
    - corrected regime : reference = the L2-normalised vector (norm = 1).
                         This is the regime you report as the main result
                         (Rimsky-aligned, unit norm x multiplier).
    - strong regime    : reference = the raw unnormalised vector (norm ~= 33).
                         This reproduces the old over-strong setting, so a
                         random control here tests whether the original large
                         effect was an over-steering artifact.
  Because this script matches whatever norm the reference file has, you do NOT
  need to know whether the apply pipeline internally normalises: real and control
  receive identical treatment, so the comparison stays fair either way.

USAGE
  Single vector:
    python make_random_baseline_vectors.py \
        --real_vector vectors/syco_en_L17_corrected.pt \
        --name syco_en_L17_corrected \
        --out_dir control_vectors \
        --n_seeds 5 --methods gaussian permute

  Batch (recommended) via a spec file:
    python make_random_baseline_vectors.py \
        --spec control_spec.json \
        --out_dir control_vectors \
        --n_seeds 5 --methods gaussian permute

  control_spec.json:
    [
      {"name": "syco_en_L17_corrected",     "real_vector": "vectors/syco_en_L17_norm.pt"},
      {"name": "syco_en_L17_strong",        "real_vector": "vectors/syco_en_L17_raw.pt"},
      {"name": "syllog_en_L20_corrected",   "real_vector": "vectors/syllog_en_L20_norm.pt"},
      {"name": "syllog_en_L20_strong",      "real_vector": "vectors/syllog_en_L20_raw.pt"}
    ]

OUTPUT
  out_dir/<name>/gaussian_seed0.pt ... gaussian_seed{N-1}.pt
  out_dir/<name>/permute_seed0.pt  ... permute_seed{N-1}.pt
  out_dir/<name>/manifest.json   (norms + cosine-sim-with-real, for verification)

Each control vector is saved as a plain torch tensor in the SAME dtype/shape as
the real vector, so it can be dropped into the existing EasyEdit apply step in
place of the real vector (point steer_vector_load_dir / the vector path at it).
"""

import argparse
import hashlib
import json
import os
import re

import torch


def derive_seed(name, method, seed):
    """Deterministic but decoupled RNG seed, so controls can never accidentally
    align with the real vector's own generation seed or with each other."""
    key = f"{name}|{method}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def derive_layer(path):
    """Pull the layer number from an EasyEdit vector filename like 'layer_17.pt'."""
    base = os.path.basename(path)
    m = re.search(r"layer_(\d+)", base)
    if not m:
        raise ValueError(
            f"Could not find 'layer_<N>' in '{base}'. EasyEdit loads layer_<N>.pt "
            f"from steer_vector_load_dir, so the real vector filename must contain it."
        )
    return int(m.group(1))


def load_vector(path, layer_key=None):
    """Load a steering vector from a .pt file.

    Handles both a plain tensor and a dict keyed by layer (some EasyEdit
    exports store {layer_idx: tensor} or {'vector': tensor}).
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        vec = obj
    elif isinstance(obj, dict):
        if layer_key is not None:
            if layer_key not in obj and str(layer_key) in obj:
                layer_key = str(layer_key)
            vec = obj[layer_key]
        elif len(obj) == 1:
            vec = next(iter(obj.values()))
        else:
            raise ValueError(
                f"{path} is a dict with keys {list(obj.keys())}; "
                f"pass --layer_key to select one."
            )
    else:
        raise TypeError(f"Unsupported object type in {path}: {type(obj)}")

    vec = vec.squeeze()
    if vec.dim() != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {tuple(vec.shape)} from {path}")
    return vec


def gaussian_random(dim, target_norm, generator, dtype):
    """Isotropic random direction rescaled to target_norm."""
    n = torch.randn(dim, generator=generator, dtype=torch.float32)
    n = n / n.norm()
    return (n * target_norm).to(dtype)


def permute_vector(vec, generator):
    """Random permutation of the vector's components (norm preserved exactly)."""
    idx = torch.randperm(vec.numel(), generator=generator)
    return vec[idx].clone()


def cosine(a, b):
    a = a.float()
    b = b.float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)


def build_for_one(name, real_path, out_root, n_seeds, methods, layer_key):
    real = load_vector(real_path, layer_key=layer_key)
    real_norm = real.norm().item()
    dim = real.numel()
    dtype = real.dtype
    layer = derive_layer(real_path)

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "name": name,
        "real_vector": os.path.abspath(real_path),
        "layer": layer,
        "dim": dim,
        "dtype": str(dtype),
        "real_norm": real_norm,
        "n_seeds": n_seeds,
        "methods": methods,
        "controls": [],
    }
    run_index = []  # one entry per control = one array-job task

    for method in methods:
        for seed in range(n_seeds):
            g = torch.Generator().manual_seed(derive_seed(name, method, seed))
            if method == "gaussian":
                v = gaussian_random(dim, real_norm, g, dtype)
            elif method == "permute":
                v = permute_vector(real, g)
            else:
                raise ValueError(f"Unknown method: {method}")

            # EasyEdit expects steer_vector_load_dir to be a FOLDER containing
            # layer_<N>.pt, so give each control its own folder.
            load_dir = os.path.join(out_dir, f"{method}_seed{seed}")
            os.makedirs(load_dir, exist_ok=True)
            torch.save(v, os.path.join(load_dir, f"layer_{layer}.pt"))

            manifest["controls"].append({
                "method": method, "seed": seed, "load_dir": os.path.abspath(load_dir),
                "norm": v.norm().item(), "cosine_with_real": cosine(v, real),
            })
            run_index.append({
                "name": name, "method": method, "seed": seed, "layer": layer,
                "load_dir": os.path.abspath(load_dir),
            })

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Verification print: norms should all match real_norm; gaussian cosine ~0.
    print(f"[{name}] layer={layer} real_norm={real_norm:.4f} dim={dim} dtype={dtype}")
    for c in manifest["controls"]:
        flag = ""
        if abs(c["norm"] - real_norm) > 1e-3 * max(1.0, real_norm):
            flag = "  <-- NORM MISMATCH!"
        print(f"    {c['method']:>8}_seed{c['seed']}  norm={c['norm']:.4f}  "
              f"cos={c['cosine_with_real']:+.4f}{flag}")
    print(f"    -> layer_{layer}.pt written under {out_dir}/<method>_seed<N>/")
    return run_index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_vector", type=str, help="Path to a single real vector .pt")
    ap.add_argument("--name", type=str, help="Label for the single-vector output subdir")
    ap.add_argument("--spec", type=str, help="JSON list of {name, real_vector} for batch mode")
    ap.add_argument("--out_dir", type=str, default="control_vectors")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["gaussian", "permute"],
                    choices=["gaussian", "permute"])
    ap.add_argument("--layer_key", type=str, default=None,
                    help="If the .pt is a dict, which key holds the vector")
    args = ap.parse_args()

    all_runs = []
    if args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
        for entry in spec:
            all_runs += build_for_one(entry["name"], entry["real_vector"], args.out_dir,
                                      args.n_seeds, args.methods, args.layer_key)
    elif args.real_vector and args.name:
        all_runs += build_for_one(args.name, args.real_vector, args.out_dir,
                                  args.n_seeds, args.methods, args.layer_key)
    else:
        ap.error("Provide either --spec, or both --real_vector and --name.")

    # One row per control = one SLURM array task.
    run_index_path = os.path.join(args.out_dir, "run_index.json")
    with open(run_index_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\n[done] {len(all_runs)} control runs indexed -> {run_index_path}")
    print(f"       submit with:  sbatch --array=0-{len(all_runs) - 1} submit_control.sh")


if __name__ == "__main__":
    main()
