#!/bin/bash
#SBATCH --job-name=translate_ko
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/translate_ko_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/translate_ko_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate translate

python3 /users/acp25mk/cross-lingual-caa/scripts/translate_dataset.py \
  --input /users/acp25mk/cross-lingual-caa/data/semeval/semeval_en_train.json \
  --output /users/acp25mk/cross-lingual-caa/data/semeval/semeval_ko_train.json \
  --src_lang eng_Latn \
  --tgt_lang kor_Hang \
  --field input
