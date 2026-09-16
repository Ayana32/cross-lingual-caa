#!/bin/bash
#SBATCH --job-name=e1_qwen35
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/e1_qwen35_%A_%a.out
#SBATCH --error=logs/e1_qwen35_%A_%a.err

source ~/.bashrc
conda activate easyedit2
cd ~/cross-lingual-caa

# 16 off-diagonal pairs: source vector -> target eval (excluding diagonal)
PAIRS=(
  "it en"
  "it zh"
  "it ko"
  "it kk"
  "zh en"
  "zh it"
  "zh ko"
  "zh kk"
  "ko en"
  "ko it"
  "ko zh"
  "ko kk"
  "kk en"
  "kk it"
  "kk zh"

  "kk ko"
)

read SRC TGT <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

VECTOR="vectors/Qwen3.5-9B/sycophancy_${SRC}/caa_vector_multiple_choice/layer_15.pt"
DATA="data/sycophancy/${TGT}/eval.json"
OUTPUT="results/qwen35_e1_${SRC}vec_${TGT}eval_m-1.5_layer15"

mkdir -p "$OUTPUT"

python scripts/evaluate_logprob_qwen35.py \
  --vector "$VECTOR" \
  --data "$DATA" \
  --layer 15 \
  --multiplier -1.5 \
  --output "$OUTPUT"
