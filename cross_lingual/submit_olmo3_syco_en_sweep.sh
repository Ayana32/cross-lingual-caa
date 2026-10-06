#!/bin/bash
for LAYER in 10 11 12 13 14 15 16 17 18 19 20 21 22 23
do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=olmo3_en_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/olmo3_syco_en_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/olmo3_syco_en_layer${LAYER}_%j.err
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

cfg = OmegaConf.load("/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_olmo3/apply_config_olmo3_layer${LAYER}.yaml")
print("Config loaded, model:", cfg.model_name_or_path)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "syco_en": json.load(open("/users/acp25mk/cross-lingual-caa/data/sycophancy/en/eval.json"))
}
vector_applier.generate(datasets, save_results=True)
print("Done. Layer ${LAYER}")
PY
SLURM
    echo "Submitted OLMo3 syco EN layer ${LAYER}"
done
echo "All OLMo3 sycophancy EN sweep submitted!"
