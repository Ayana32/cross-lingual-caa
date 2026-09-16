#!/bin/bash
#SBATCH --job-name=g4_native_en
#SBATCH --output=logs/gemma4_native_en_%j.out
#SBATCH --error=logs/gemma4_native_en_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH
cd ~/cross-lingual-caa

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_gemma4/apply_config_gemma4_en_m-1.0_layer26.yaml")
vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()
datasets = {"syco_en": json.load(open("data/sycophancy/en/eval.json"))}
vector_applier.generate(datasets, save_results=True)
print("Done. Gemma4 native EN layer26")
PY
