import json, random

with open('sycophancy_raw.json') as f:
    raw = json.load(f)

random.seed(42)
random.shuffle(raw)

# EasyEdit2 train format
train = [{"question": d["question"], "matching": d["answer_matching_behavior"], "not_matching": d["answer_not_matching_behavior"]} for d in raw[:100]]

# Generation test set (EN)
gen_en = [{"input": d["question"]} for d in raw[100:150]]

with open('train_en.json', 'w') as f:
    json.dump(train, f, indent=2)
with open('gen_en.json', 'w') as f:
    json.dump(gen_en, f, indent=2)

print(f"Train: {len(train)} pairs")
print(f"Gen EN: {len(gen_en)} prompts")
print("Sample:", train[0])
