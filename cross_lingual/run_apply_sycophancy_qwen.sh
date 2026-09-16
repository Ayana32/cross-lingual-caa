#!/bin/bash
#SBATCH --job-name=apply_caa_qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/acp25mk/EasyEdit/logs/apply_caa_qwen_%j.out
#SBATCH --error=/users/acp25mk/EasyEdit/logs/apply_caa_qwen_%j.err

set -euo pipefail

echo "=== DEBUG ==="
date
hostname
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
echo "============="

cd /users/acp25mk/EasyEdit


# --- Conda init (system Anaconda on Stanage) ---
source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available());"

python3 -u - <<'PY'
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

CFG_PATH = "/users/acp25mk/EasyEdit/hparams/Steer/experiment_hparams/spilt_experiment/apply_config_sycophancy_qwen.yaml"
EN_PATH  = "/users/acp25mk/EasyEdit/data/gen_en_test.json"
KO_PATH  = "/users/acp25mk/EasyEdit/data/gen_ko_test.json"

cfg = OmegaConf.load(CFG_PATH)
print("Loaded cfg:", CFG_PATH)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()
print("Vectors applied.")

en = json.load(open(EN_PATH, "r", encoding="utf-8"))
ko = json.load(open(KO_PATH, "r", encoding="utf-8"))

datasets = {
    "gen_en_test": en,
    "gen_ko_test": ko,
}

print("Datasets loaded:", {k: len(v) for k, v in datasets.items()})

vector_applier.generate(datasets, save_results=True)
print("Done. Output dir:", cfg.generation_output_dir)
PY
