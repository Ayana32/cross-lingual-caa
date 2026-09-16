#!/bin/bash
#SBATCH --job-name=tqa_freegen
#SBATCH --array=0-1
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/tqa_freegen_%A_%a.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/tqa_freegen_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

LANGS=(ko kk)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

python scripts/generate_freegen_tqa.py \
    --data data/truthfulqa/${LANG}/eval_freegen.json \
    --vector vectors/Qwen3.5-9B/truthfulqa/caa_vector/truthfulqa_en/caa_vector_multiple_choice/layer_15.pt \
    --layer 15 \
    --multiplier -1.5 \
    --output results/tqa_freegen_qwen35_${LANG}_layer15
