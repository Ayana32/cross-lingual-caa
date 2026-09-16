#!/bin/bash
for LANG in it kk ko zh; do
for LAYER in 13 14 15 16 17 18; do

    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=q35_${LANG}_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/qwen35_syco_${LANG}_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/qwen35_syco_${LANG}_layer${LAYER}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:\${PYTHONPATH:-}
cd /users/acp25mk/cross-lingual-caa

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment/apply_configs_${LANG}/apply_config_qwen35_${LANG}_layer${LAYER}.yaml")
print("Config loaded:", cfg.model_name_or_path, "lang=${LANG} layer=${LAYER}")

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "syco_${LANG}": json.load(open("/users/acp25mk/cross-lingual-caa/data/sycophancy/${LANG}/eval.json"))
}
vector_applier.generate(datasets, save_results=True)
print("Done. Lang=${LANG} Layer=${LAYER}")
PY
SLURM
    echo "Submitted ${LANG} layer ${LAYER}"
done
done
echo "All submitted!"
