#!/bin/bash
#SBATCH --job-name=translate_kk
#SBATCH --output=/users/acp25mk/cross-lingual-caa/logs/translate_kk_final_%j.out
#SBATCH --error=/users/acp25mk/cross-lingual-caa/logs/translate_kk_final_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source /opt/apps/testapps/common/software/staging/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate easyedit2

python3 - << 'PYEOF'
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_path = "/users/acp25mk/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
print("Loading model from local cache...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
print("Model loaded!")

data = json.load(open("/users/acp25mk/cross-lingual-caa/data/sycophancy_en_train.json"))
tokenizer.src_lang = "eng_Latn"
target_id = tokenizer.convert_tokens_to_ids("kaz_Cyrl")

results = []
batch_size = 8
texts = [item["input"] for item in data]

print(f"Translating {len(texts)} items...")
for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    inputs = tokenizer(
        batch, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=512
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for item, translation in zip(data[i:i+batch_size], decoded):
        new_item = item.copy()
        new_item["input"] = translation
        new_item["question"] = translation
        results.append(new_item)

output_path = "/users/acp25mk/cross-lingual-caa/data/sycophancy_kk_train.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done! Saved {len(results)} items to {output_path}")
PYEOF
