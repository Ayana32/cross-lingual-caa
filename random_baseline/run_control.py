#!/usr/bin/env python3
import argparse
import json
import os
import sys

from omegaconf import OmegaConf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--qwen35_config", required=True)
    ap.add_argument("--gemma4_config", required=True)
    args = ap.parse_args()

    runs = json.load(open(args.manifest))
    if args.task_id < 0 or args.task_id >= len(runs):
        raise IndexError(f"task_id {args.task_id} out of range 0..{len(runs)-1}")
    run = runs[args.task_id]

    # 모델별 base config 선택
    if run["name"].startswith("qwen35"):
        base_config_path = args.qwen35_config
    else:
        base_config_path = args.gemma4_config

    cfg = OmegaConf.load(base_config_path)

    # PLACEHOLDER 채우기
    cfg.steer_vector_load_dir = [run["load_dir"]]
    cfg.generation_output_dir = os.path.join(
        args.output_root, run["name"],
        f"{run['method']}_seed{run['seed']}"
    )
    os.makedirs(cfg.generation_output_dir, exist_ok=True)

    print(f"[task {args.task_id}] {run['name']} / {run['method']}_seed{run['seed']}")
    print(f"  model: {cfg.model_name_or_path}")
    print(f"  steer_vector_load_dir = {run['load_dir']}")
    print(f"  generation_output_dir = {cfg.generation_output_dir}")

    sys.path.insert(0, "/users/acp25mk/EasyEdit2")
    from steer.vector_appliers.vector_applier import BaseVectorApplier

    vector_applier = BaseVectorApplier(cfg)
    vector_applier.apply_vectors()

    datasets = {}
    for data_path in cfg.generation_data:
        parts = data_path.rstrip("/").split("/")
        key = parts[-2] + "_" + os.path.splitext(parts[-1])[0]
        datasets[key] = json.load(open(data_path))
        print(f"  dataset: {key} ({len(datasets[key])} items)")

    vector_applier.generate(datasets, save_results=True)
    print("Done.")

if __name__ == "__main__":
    main()
