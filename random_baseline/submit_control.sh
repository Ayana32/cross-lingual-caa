#!/bin/bash
#SBATCH --job-name=caa_control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/control_%A_%a.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/control_%A_%a.err

set -euo pipefail

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:${PYTHONPATH:-}

cd /users/acp25mk/cross-lingual-caa

python random_baseline/run_control.py \
    --manifest random_baseline/control_vectors/run_index.json \
    --output_root results/random_baseline \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --qwen35_config random_baseline/configs/base_config_qwen35_layer15.yaml \
    --gemma4_config random_baseline/configs/base_config_gemma4_layer26.yaml
