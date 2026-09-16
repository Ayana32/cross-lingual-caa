"""
PCA-based steering vector extraction.
Replaces CAA mean-difference with PCA on per-sample diffs.
"""
import os
import sys
sys.path.append("/users/acp25mk/EasyEdit")

import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from omegaconf import OmegaConf

from steer.models.get_model import get_model
from steer.datasets.caa_data import get_tokens_for_caa
from steer.vector_generators.caa.generate_caa_hparam import CAAHyperParams

def generate_pca_vectors(config_path, config_name, output_suffix="pca_vector_multiple_choice"):
    cfg_file = Path(config_path) / f"{config_name}.yaml"
    cfg = OmegaConf.load(cfg_file)

    # Load hparams
    hparam_path = "/users/acp25mk/cross-lingual-caa/hparams/Steer/caa_hparams/generate_pca_layers12_26.yaml"
    hparams = CAAHyperParams.from_hparams(hparam_path)
    hparams.model_name_or_path = cfg.model_name_or_path
    hparams.torch_dtype = cfg.torch_dtype
    hparams.device = cfg.device
    hparams.use_chat_template = cfg.use_chat_template
    hparams.system_prompt = cfg.system_prompt

    # Load dataset
    from steer.datasets.dataset_loader import DatasetLoader
    dataset_name = cfg.steer_train_dataset[0]
    loader = DatasetLoader()
    dataset = loader.load_file(dataset_name, split='train')

    # Load model
    model, tokenizer = get_model(hparams)

    # Collect per-sample diffs
    diff_activations = dict([(layer, []) for layer in hparams.layers])

    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

    for p_dict, n_dict in tqdm(zip(pos_tokens_list, neg_tokens_list), total=len(pos_tokens_list)):
        p_tokens = p_dict["pos_tokens"]
        n_tokens = n_dict["neg_tokens"]

        model.reset_all()
        model.get_logits(p_tokens)
        p_acts = {layer: model.get_last_activations(layer)[0, -2, :].detach().cpu() for layer in hparams.layers}

        model.reset_all()
        model.get_logits(n_tokens)
        n_acts = {layer: model.get_last_activations(layer)[0, -2, :].detach().cpu() for layer in hparams.layers}

        for layer in hparams.layers:
            diff_activations[layer].append(p_acts[layer] - n_acts[layer])

    # PCA extraction
    output_dir = Path(cfg.steer_vector_output_dirs[0]) / output_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in hparams.layers:
        diffs = torch.stack(diff_activations[layer]).float().numpy()  # [N, hidden_dim]
        pca = PCA(n_components=1)
        pca.fit(diffs)
        vec = pca.components_[0]
        vec = vec / np.linalg.norm(vec)  # normalize
        torch.save(torch.tensor(vec, dtype=torch.float32), output_dir / f"layer_{layer}.pt")
        print(f"Layer {layer}: saved PCA vector (explained variance: {pca.explained_variance_ratio_[0]:.4f})")

    print(f"Done! Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args()
    generate_pca_vectors(args.config_path, args.config_name)
