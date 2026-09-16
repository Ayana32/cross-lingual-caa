#!/bin/bash
#SBATCH --job-name=test_olmo3_wrapper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --qos=gpu
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/test_olmo3_wrapper_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/test_olmo3_wrapper_%j.err

export PATH=/users/acp25mk/.conda/envs/easyedit2/bin:$PATH
export PYTHONPATH=/users/acp25mk/EasyEdit2:$PYTHONPATH

echo "NODE: $(hostname)"
echo "PYTHON: $(which python)"

python - <<'PY'
import torch
from types import SimpleNamespace
from steer.models.get_model import get_model

hparams = SimpleNamespace(
    model_name_or_path="allenai/OLMo-3-7B-Instruct",
    dtype="bfloat16",
    torch_dtype=None,
    use_cache=True,
    use_chat_template=False,
    device="cuda",
    override_model_weights_path=None,
    vllm_enable=False,
    save_activations=True,
)

print("Loading OLMo3...")

model, tokenizer = get_model(hparams)

print("\n=== WRAPPER TEST ===")
print("wrapper class:", type(model).__name__)
print("HF model class:", type(model.model).__name__)

layers = model._decoder_layers()
norm = model._final_norm()
lm_head = model._lm_head()

print("num decoder layers:", len(layers))
print("first wrapped layer:", type(layers[0]).__name__)
print("last wrapped layer:", type(layers[-1]).__name__)
print("final norm:", type(norm).__name__)
print("lm head:", type(lm_head).__name__)

print("\n=== EXPECTATIONS ===")
print("32 layers:", len(layers) == 32)
print("CUDA available:", torch.cuda.is_available())
print("model device:", next(model.model.parameters()).device)

print("\nOLMo3 wrapper smoke test PASSED")
PY
