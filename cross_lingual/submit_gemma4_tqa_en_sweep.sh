#!/bin/bash
for LAYER in 20 21 22 23 24 25 26 27 28 29 30 31 32 33; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=tqa_g4_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/tqa_gemma4_en_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/tqa_gemma4_en_layer${LAYER}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:\$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:\$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_tqa_gemma4/apply_config_tqa_gemma4_layer${LAYER}.yaml")
vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()
datasets = {"tqa_en": json.load(open("data/truthfulqa/en/eval.json"))}
vector_applier.generate(datasets, save_results=True)
print("Done. Gemma4 TQA EN Layer ${LAYER}")
PY
SLURM
    echo "Submitted Gemma4 TQA EN layer ${LAYER}"
done
echo "All Gemma4 TQA EN sweep jobs submitted!"
