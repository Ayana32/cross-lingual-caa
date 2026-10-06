#!/bin/bash
for LAYER in 20 21 22 23 24 25 26 27 28 29 30 31 32 33
do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=g4_en_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/gemma4_syco_en_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/gemma4_syco_en_layer${LAYER}_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:\$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_gemma4/apply_config_gemma4_layer${LAYER}.yaml")
print("Config loaded, model:", cfg.model_name_or_path)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "syco_en": json.load(open("/users/acp25mk/cross-lingual-caa/data/sycophancy_en_train.json"))
}
vector_applier.generate(datasets, save_results=True)
print("Done. Layer ${LAYER}")
PY
SLURM
    echo "Submitted Gemma4 syco EN layer ${LAYER}"
done
echo "All Gemma4 sycophancy EN sweep submitted!"
