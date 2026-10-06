#!/usr/bin/env python3
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def translate_batch(texts, tokenizer, model, src_lang, tgt_lang, device, batch_size=8):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i+batch_size]
        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        target_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_length=512
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--src_lang", default="eng_Latn")
    parser.add_argument("--tgt_lang", required=True)
    parser.add_argument("--field", default="input")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading NLLB model...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype="auto",
        use_safetensors=True,
    )

    with open(args.input) as f:
        data = json.load(f)

    texts = [item[args.field] for item in data]
    print(f"Translating {len(texts)} items: {args.src_lang} -> {args.tgt_lang}")

    translated = translate_batch(texts, tokenizer, model, args.src_lang, args.tgt_lang, device, args.batch_size)

    for item, trans in zip(data, translated):
        item[args.field] = trans
        item['question'] = trans

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} items to {args.output}")

if __name__ == "__main__":
    main()
