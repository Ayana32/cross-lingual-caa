#!/bin/bash
#SBATCH --job-name=logprob_test_en
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/logprob_test_en_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/logprob_test_en_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

python /users/acp25mk/cross-lingual-caa/scripts/evaluate_logprob.py \
  --data /users/acp25mk/cross-lingual-caa/data/sycophancy_en_train.json \
  --vector /users/acp25mk/cross-lingual-caa/vectors/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice/layer_17.pt \
  --layer 17 \
  --multiplier 1.5 \
  --output /users/acp25mk/cross-lingual-caa/results/logprob_test/en_l17_m1.5.json
