#!/bin/bash
for LANG in en ko es kk; do
for LAYER in 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=q35_sv_${LANG}_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/qwen35_semeval_${LANG}_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/qwen35_semeval_${LANG}_layer${LAYER}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:\$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.load("/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/semeval_experiment/apply_configs_qwen35/apply_config_qwen35_${LANG}_layer${LAYER}.yaml")
print("Config loaded, model:", cfg.model_name_or_path)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "semeval_${LANG}": json.load(open("/users/acp25mk/cross-lingual-caa/data/semeval/semeval_${LANG}_train.json"))
}
vector_applier.generate(datasets, save_results=True)
print("Done. Lang ${LANG} Layer ${LAYER}")
PY
SLURM
    echo "Submitted Qwen3.5 semeval ${LANG} layer ${LAYER}"
done
done
echo "All semeval sweep submitted!"
