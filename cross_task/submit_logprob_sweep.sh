#!/bin/bash
BASE=/users/acp25mk/cross-lingual-caa

declare -A SYCO_DATA=(
    [EN]="$BASE/data/sycophancy_en_train.json"
    [KO]="$BASE/data/sycophancy_ko_train.json"
    [ES]="$BASE/data/sycophancy_es_train.json"
    [KK]="$BASE/data/sycophancy_kk_train.json"
)

declare -A UNNORM_VEC=(
    [EN]="$BASE/vectors/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice"
    [KO]="$BASE/vectors/Qwen2.5-7B-Instruct-ko/sycophancy_ko/caa_vector_multiple_choice"
    [ES]="$BASE/vectors/Qwen2.5-7B-Instruct-es/sycophancy_es/caa_vector_multiple_choice"
    [KK]="$BASE/vectors/Qwen2.5-7B-Instruct-kk/sycophancy_kk/caa_vector_multiple_choice"
)

declare -A NORM_VEC=(
    [EN]="$BASE/vectors/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice_normalized"
    [KO]="$BASE/vectors/Qwen2.5-7B-Instruct-ko/sycophancy_ko/caa_vector_multiple_choice_normalized"
    [ES]="$BASE/vectors/Qwen2.5-7B-Instruct-es/sycophancy_es/caa_vector_multiple_choice_normalized"
    [KK]="$BASE/vectors/Qwen2.5-7B-Instruct-kk/sycophancy_kk/caa_vector_multiple_choice_normalized"
)

mkdir -p "$BASE/logs"

for LANG in EN KO ES KK; do
    for VECTYPE in unnorm norm; do
        for MULT in 1.0 1.5 2.0; do

            if [ "$VECTYPE" = "unnorm" ]; then
                VEC_DIR=${UNNORM_VEC[$LANG]}
            else
                VEC_DIR=${NORM_VEC[$LANG]}
            fi

            DATA=${SYCO_DATA[$LANG]}
            JOB_NAME="logprob_${VECTYPE}_${LANG,,}_m${MULT}"
            OUT_DIR="$BASE/results/logprob_sweep/${VECTYPE}_${LANG,,}_m${MULT}"

            mkdir -p "$OUT_DIR"

            if [ ! -f "$DATA" ]; then
                echo "Missing data: $DATA — skipping"
                continue
            fi

            sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$BASE/logs/${JOB_NAME}_%j.out
#SBATCH --error=$BASE/logs/${JOB_NAME}_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

for LAYER in 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26; do
    VEC_FILE="${VEC_DIR}/layer_\${LAYER}.pt"
    if [ ! -f "\$VEC_FILE" ]; then
        echo "Missing vector: \$VEC_FILE — skipping"
        continue
    fi
    python /users/acp25mk/cross-lingual-caa/scripts/evaluate_logprob.py \
        --data ${DATA} \
        --vector \${VEC_FILE} \
        --layer \${LAYER} \
        --multiplier ${MULT} \
        $([ "${VECTYPE}" = "norm" ] && echo "--normalize") \
        --output ${OUT_DIR}/layer_\${LAYER}_m${MULT}.json
    echo "Done: ${VECTYPE} ${LANG} layer \${LAYER} m=${MULT}"
done
echo "All layers done: ${VECTYPE} ${LANG} m=${MULT}"
SLURM

            echo "Submitted: $VECTYPE $LANG m=$MULT"
        done
    done
done
echo "All 24 jobs submitted!"
