#!/bin/bash
#SBATCH --job-name=qwen35_en_row
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/qwen35_en_row_%A_%a.out
#SBATCH --error=logs/qwen35_en_row_%A_%a.err

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH
cd ~/cross-lingual-caa

PAIRS=("en en" "en it" "en zh" "en ko" "en kk")
read SRC TGT <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

python scripts/evaluate_logprob_qwen35.py \
  --vector vectors/Qwen3.5-9B/sycophancy/caa_vector_multiple_choice/layer_15.pt \
  --data data/sycophancy/${TGT}/eval.json \
  --layer 15 \
  --multiplier -1.5 \
  --output results/qwen35_e1_${SRC}vec_${TGT}eval_m-1.5_layer15/results.json
