#!/bin/bash
for LANG in it zh ko kk; do
  for LAYER in 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=tqa_o3_${LANG}_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/tqa_olmo3_${LANG}_m1.5_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/tqa_olmo3_${LANG}_m1.5_layer${LAYER}_%j.err
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

cfg = OmegaConf.load("hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_tqa_olmo3_multilang/apply_config_tqa_olmo3_${LANG}_layer${LAYER}.yaml")
vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()
datasets = {"tqa_${LANG}": json.load(open("data/truthfulqa/${LANG}/eval.json"))}
vector_applier.generate(datasets, save_results=True)
print("Done. OLMo3 TQA ${LANG} Layer ${LAYER}")
PY
SLURM
    echo "Submitted OLMo3 TQA ${LANG} layer ${LAYER}"
  done
done
echo "All OLMo3 TQA multilang sweep submitted!"
