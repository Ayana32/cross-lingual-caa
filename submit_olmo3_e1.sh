#!/bin/bash
#SBATCH --job-name=olmo3_e1
#SBATCH --array=0-19
#SBATCH --output=logs/olmo3_e1_%A_%a.out
#SBATCH --error=logs/olmo3_e1_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH
cd ~/cross-lingual-caa

CONFIGS=($(ls hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_olmo3_e1/*.yaml | sort))

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Running: $CFG"

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("$CFG")
vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()
tgt = "$CFG".split("eval.yaml")[0].split("_")[-1]
datasets = {"syco_" + tgt: json.load(open(f"data/sycophancy/{tgt}/eval.json"))}
vector_applier.generate(datasets, save_results=True)
print("Done:", "$CFG")
PY
