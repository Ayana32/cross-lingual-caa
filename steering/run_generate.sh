#!/bin/bash
#SBATCH --job-name=caa_generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load Anaconda3/2022.05
module load GCC/12.2.0
source activate easyedit2

cd /users/acp25mk/EasyEdit

# Check GPU
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Generate steering vectors
python vectors_generate.py --config-path hparams/Steer/experiment_hparams/sycophancy_experiment --config-name config
