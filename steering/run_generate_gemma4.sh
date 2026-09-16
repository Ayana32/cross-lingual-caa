#!/bin/bash
#SBATCH --job-name=gemma4_generate
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/gemma4_generate_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/gemma4_generate_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

HF_HUB_DISABLE_XET=1 python steering/vectors_generate.py \
    --config-path /users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment \
    --config-name config_gemma4
