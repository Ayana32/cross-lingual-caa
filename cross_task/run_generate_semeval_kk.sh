#!/bin/bash
#SBATCH --job-name=gen_semeval_kk
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/generate_semeval_kk_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/generate_semeval_kk_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
cd /users/acp25mk/EasyEdit

python vectors_generate.py \
  --config-path /users/acp25mk/cross-lingual-caa/hparams/sycophancy_experiment \
  --config-name config_semeval_kk
