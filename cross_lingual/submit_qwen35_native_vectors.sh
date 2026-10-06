#!/bin/bash

for LANG in it kk ko zh; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=q35_vec_${LANG}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/q35_vec_${LANG}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/q35_vec_${LANG}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

export PYTHONPATH=/users/acp25mk/EasyEdit2:\${PYTHONPATH:-}

cd /users/acp25mk/cross-lingual-caa

python steering/vectors_generate.py \
  --config-path /users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment \
  --config-name config_qwen35_${LANG}

SLURM

    echo "Submitted ${LANG}"
done
