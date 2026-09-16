#!/bin/bash
for LAYER in 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=sweep_en_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/sweep_en_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/sweep_en_layer${LAYER}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
cd /users/acp25mk/EasyEdit

python3 -u - << PY
import json
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

CFG_PATH = "/users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/spilt_experiment/apply_config_en_layer${LAYER}.yaml"
cfg = OmegaConf.load(CFG_PATH)
print("Loaded cfg:", CFG_PATH)

vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

datasets = {
    "semeval_en_layer${LAYER}": json.load(open("/users/acp25mk/cross-lingual-caa/data/semeval/semeval_en_train.json"))
}
print("Datasets loaded:", {k: len(v) for k, v in datasets.items()})
vector_applier.generate(datasets, save_results=True)
print("Done. Output dir:", cfg.generation_output_dir)
PY
SLURM

    echo "Submitted layer ${LAYER}"
done
echo "All layers submitted!"
