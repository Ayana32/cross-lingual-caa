#!/bin/bash
for LANG in it zh ko kk; do
  for LAYER in 12 13 14 15 16; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=o3_${LANG}_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/olmo3_syco_${LANG}_m-1.5_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/olmo3_syco_${LANG}_m-1.5_layer${LAYER}_%j.err
#SBATCH --time=05:00:00
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

cfg = OmegaConf.load("hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_olmo3_multilang/apply_config_olmo3_${LANG}_m-1.5_layer${LAYER}.yaml")
print("Config loaded, model:", cfg.model_name_or_path)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {"syco_${LANG}": json.load(open("data/sycophancy/${LANG}/eval.json"))}
vector_applier.generate(datasets, save_results=True)
print("Done. OLMo3 ${LANG} Layer ${LAYER}")
PY
SLURM
    echo "Submitted OLMo3 syco ${LANG} layer ${LAYER}"
  done
done
echo "All submitted!"
