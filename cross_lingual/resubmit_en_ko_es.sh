#!/bin/bash
LANGS=("en" "ko" "es" "kk")
MULTIPLIERS=("-1.5" "-1.0" "0.0" "1.0" "1.5")
VECTOR_DIRS=(
    "/users/acp25mk/cross-lingual-caa/vectors/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice_normalized"
    "/users/acp25mk/cross-lingual-caa/vectors/Qwen2.5-7B-Instruct-ko/sycophancy_ko/caa_vector_multiple_choice_normalized"
    "/users/acp25mk/cross-lingual-caa/vectors/Qwen2.5-7B-Instruct-es/sycophancy_es/caa_vector_multiple_choice_normalized"
    "/users/acp25mk/EasyEdit/vectors/Qwen2.5-7B-Instruct-kk/sycophancy_kk/caa_vector_multiple_choice_normalized"
)
DATA_FILES=(
    "/users/acp25mk/cross-lingual-caa/data/sycophancy_en_train.json"
    "/users/acp25mk/cross-lingual-caa/data/sycophancy_ko_train.json"
    "/users/acp25mk/cross-lingual-caa/data/sycophancy_es_train.json"
    "/users/acp25mk/cross-lingual-caa/data/sycophancy_kk_train.json"
)
for i in "${!LANGS[@]}"; do
    LANG=${LANGS[$i]}
    VDIR=${VECTOR_DIRS[$i]}
    DFILE=${DATA_FILES[$i]}
    for LAYER in 15 16 17 18 19 20 21; do
        for MULT in "${MULTIPLIERS[@]}"; do
            RESULT_DIR="/users/acp25mk/cross-lingual-caa/results/norm_${LANG}_m${MULT}_layer${LAYER}"
            if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR 2>/dev/null)" ]; then
                echo "Skip: ${LANG} layer${LAYER} mult${MULT}"
                continue
            fi
            MULT_TAG=$(echo $MULT | sed 's/-/neg/g' | sed 's/\./_/g')
            sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=norm_${LANG}_${MULT_TAG}_l${LAYER}
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/norm_${LANG}_m${MULT}_layer${LAYER}_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/norm_${LANG}_m${MULT}_layer${LAYER}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2
cd /users/acp25mk/EasyEdit

python3 -u - << PY
import json, yaml
from omegaconf import OmegaConf
from steer.vector_appliers.vector_applier import BaseVectorApplier

cfg = OmegaConf.create({
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "torch_dtype": "bfloat16",
    "device": "cuda:0",
    "seed": 42,
    "use_chat_template": True,
    "system_prompt": "",
    "apply_steer_hparam_paths": ["/tmp/apply_norm_${LANG}_${LAYER}_${MULT_TAG}.yaml"],
    "steer_vector_load_dir": ["${VDIR}"],
    "generation_data": ["${DFILE}"],
    "generation_data_size": None,
    "generation_output_dir": "/users/acp25mk/cross-lingual-caa/results/norm_${LANG}_m${MULT}_layer${LAYER}",
    "num_responses": 1,
    "steer_from_end_position": False,
    "generate_orig_output": True,
    "generation_params": {"max_new_tokens": 128, "do_sample": False, "temperature": 0.0}
})

hparam_content = {"alg_name": "caa", "layers": [${LAYER}], "multipliers": [${MULT}], "save_activations": False, "steer_vector_load_dir": "${VDIR}"}
hparam_path = "/tmp/apply_norm_${LANG}_${LAYER}_${MULT_TAG}.yaml"
with open(hparam_path, "w") as f:
    yaml.dump(hparam_content, f)

cfg.apply_steer_hparam_paths = [hparam_path]
vector_applier = BaseVectorApplier(cfg)
vector_applier.apply_vectors()

with open("${DFILE}") as f:
    dataset = json.load(f)
datasets = {"syco_${LANG}_norm_m${MULT}_layer${LAYER}": dataset}
vector_applier.generate(datasets, save_results=True)
print("Done.")
PY
SLURM
            echo "Submitted: ${LANG} layer${LAYER} mult${MULT}"
        done
    done
done
echo "All jobs submitted!"
