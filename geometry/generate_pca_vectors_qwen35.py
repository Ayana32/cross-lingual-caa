"""
PCA-based steering vector extraction for Qwen3.5-9B.
Supervisor-recommended method (Panickssery et al.)
Uses EasyEdit2 infrastructure identical to generate_caa_vectors.py
"""
import sys
sys.path.insert(0, "/users/acp25mk/EasyEdit2")

import os
import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from omegaconf import OmegaConf

def generate_pca_vectors(config_path, config_name, layers=list(range(13, 19))):
    from steer.vector_generators.caa.generate_caa_hparam import CAAHyperParams
    from steer.models.get_model import get_model
    from steer.datasets.caa_data import get_tokens_for_caa
    from steer.datasets.dataset_loader import DatasetLoader

    # Load top config
    cfg_file = Path(config_path) / f"{config_name}.yaml"
    cfg = OmegaConf.load(cfg_file)

    # Build hparams
    hparams = CAAHyperParams(
        model_name_or_path=cfg.model_name_or_path,
        torch_dtype=cfg.torch_dtype,
        device=cfg.device,
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        layers=layers,
        alg_name="caa",
        multiple_choice=True,
        save_activations=True,
        save_vectors=False,
        seed=42,
    )

    # Load dataset
    dataset_name = cfg.steer_train_dataset[0]
    loader = DatasetLoader()
    dataset = loader.load_file(dataset_name, split='train')
    print(f"Loaded {len(dataset)} samples ({dataset_name})")

    # Load model
    model, tokenizer = get_model(hparams)

    # Collect activations (same as generate_caa_vectors.py)
    pos_activations = {layer: [] for layer in layers}
    neg_activations = {layer: [] for layer in layers}

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_dict, n_dict in tqdm(zip(pos_tokens_list, neg_tokens_list), total=len(pos_tokens_list)):
        p_tokens = p_dict["pos_tokens"]
        n_tokens = n_dict["neg_tokens"]

        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_act = model.get_last_activations(layer)[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_act)

        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_act = model.get_last_activations(layer)[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_act)

    # PCA extraction (교수님 방법)
    lang = dataset_name.replace("sycophancy_", "")
    output_dir = Path(cfg.steer_vector_output_dirs[0]) / f"{dataset_name}/pca_vector_multiple_choice"
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        all_pos = torch.stack(pos_activations[layer]).float().numpy()
        all_neg = torch.stack(neg_activations[layer]).float().numpy()
        diffs = all_pos - all_neg  # [N, hidden_dim]

        pca = PCA(n_components=1)
        pca.fit(diffs)
        vec = pca.components_[0]
        vec = vec / np.linalg.norm(vec)

        out_path = output_dir / f"layer_{layer}.pt"
        torch.save(torch.tensor(vec, dtype=torch.float32), out_path)
        print(f"  Layer {layer}: explained_var={pca.explained_variance_ratio_[0]:.4f} → {out_path}")

    print(f"\nDone! Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args()
    generate_pca_vectors(args.config_path, args.config_name)
