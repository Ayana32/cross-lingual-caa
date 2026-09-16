#!/bin/bash
#SBATCH --job-name=pca_generate_es
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/generate_pca_es_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/generate_pca_es_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

cd /users/acp25mk/EasyEdit

python /users/acp25mk/cross-lingual-caa/scripts/generate_pca_vectors.py \
  --config-path /users/acp25mk/cross-lingual-caa/hparams/sycophancy_experiment \
  --config-name config_es
