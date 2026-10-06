#!/bin/bash
# Native TQA vector extraction: 3 models x 4 target languages = 12 jobs.
# Environment/invocation copied from steering/run_generate_truthfulqa.sh (EN TQA),
# with the language loop from cross_lingual/submit_qwen35_native_vectors.sh.
#
# PREREQUISITE: run make_tqa_native_configs.sh first.
# Run from repo root: bash submit_tqa_native_vectors.sh

for MODEL in qwen35 gemma4 olmo3; do
  for LANG in it zh ko kk; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=tqa_nat_${MODEL}_${LANG}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/tqa_nat_${MODEL}_${LANG}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/tqa_nat_${MODEL}_${LANG}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:\$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:\$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

HF_HUB_DISABLE_XET=1 python steering/vectors_generate.py \\
    --config-path /users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment \\
    --config-name config_${MODEL}_truthfulqa_${LANG}
SLURM
    echo "Submitted native TQA vector generation: ${MODEL} ${LANG}"
  done
done
