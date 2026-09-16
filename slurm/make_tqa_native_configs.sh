#!/bin/bash
# Generate per-language native TQA vector configs from each model's EN config.
# Only the steer_train_dataset name is swapped (truthfulqa_en -> truthfulqa_<lang>).
# Output dir is inherited unchanged; vectors auto-nest under truthfulqa_<lang>/.
#
# Run from the repo root: bash make_tqa_native_configs.sh

set -euo pipefail

CFG_DIR="hparams/Steer/experiment_hparams/sycophancy_experiment"
MODELS=(qwen35 gemma4 olmo3)
LANGS=(it zh ko kk)

for m in "${MODELS[@]}"; do
  EN="${CFG_DIR}/config_${m}_truthfulqa.yaml"
  if [[ ! -f "$EN" ]]; then
    echo "MISSING: $EN  (skipping ${m})" >&2
    continue
  fi
  for lang in "${LANGS[@]}"; do
    OUT="${CFG_DIR}/config_${m}_truthfulqa_${lang}.yaml"
    sed "s/truthfulqa_en/truthfulqa_${lang}/g" "$EN" > "$OUT"
    echo "wrote $OUT"
  done
done

echo
echo "Sanity check (diff EN vs IT for qwen35, should differ only on the dataset line):"
diff "${CFG_DIR}/config_qwen35_truthfulqa.yaml" "${CFG_DIR}/config_qwen35_truthfulqa_it.yaml" || true
