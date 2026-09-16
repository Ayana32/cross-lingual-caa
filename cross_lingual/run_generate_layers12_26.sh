#!/bin/bash
#SBATCH --job-name=caa_generate_12_26
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/generate_12_26_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/generate_12_26_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

cd /users/acp25mk/EasyEdit

python vectors_generate.py \
  --config-path /users/acp25mk/cross-lingual-caa/hparams/sycophancy_experiment \
  --config-name config_layers12_26
