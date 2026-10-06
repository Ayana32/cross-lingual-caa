#!/bin/bash
for LANG in it zh ko kk; do
    sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=o3_vec_${LANG}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/olmo3_vec_${LANG}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/olmo3_vec_${LANG}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:\$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:\$PYTHONPATH
cd /users/acp25mk/cross-lingual-caa

HF_HUB_DISABLE_XET=1 python steering/vectors_generate.py \
    --config-path /users/acp25mk/cross-lingual-caa/hparams/Steer/experiment_hparams/sycophancy_experiment \
    --config-name config_olmo3_${LANG}
SLURM
    echo "Submitted OLMo3 native vector ${LANG}"
done
