#!/bin/bash
#SBATCH --job-name=apply_caa_semeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/apply_caa_semeval_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/apply_caa_semeval_%j.err

set -euo pipefail

echo "=== DEBUG ==="
date
hostname
echo "PWD: $(pwd)"
nvidia-smi || true
echo "============="

cd /users/acp25mk/EasyEdit

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"

python3 -u - <<'PY'
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

CFG_PATH = "/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/spilt_experiment/apply_config_semeval_en.yaml"

cfg = OmegaConf.load(CFG_PATH)
print("Loaded cfg:", CFG_PATH)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "semeval_en_train": json.load(open("/users/acp25mk/cross-lingual-caa/data/semeval/semeval_en_train.json"))
}

print("Datasets loaded:", {k: len(v) for k, v in datasets.items()})
vector_applier.generate(datasets, save_results=True)
print("Done. Output dir:", cfg.generation_output_dir)
PY
