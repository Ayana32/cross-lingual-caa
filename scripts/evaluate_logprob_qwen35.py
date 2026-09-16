"""
Log-probability based evaluation for CAA steering.
Replaces generation + parsing with direct P(A) vs P(B) comparison.
Adapted for Qwen3.5-9B hybrid architecture.
"""
import sys
sys.path.append("/users/acp25mk/EasyEdit2")

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def get_ab_logprobs(model, tokenizer, input_text, token_a, token_b, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[token_a].item(), log_probs[token_b].item()


def evaluate(data_path, vector_path, layer, multiplier, output_path, normalize=False, device="cuda:0"):
    print("Loading model...")
    model_name = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Qwen3.5 token IDs for (A and (B
    token_a = tokenizer.encode("(A)", add_special_tokens=False)[0]  # 4205
    token_b = tokenizer.encode("(B)", add_special_tokens=False)[0]  # 5181
    print(f"Token IDs — (A): {token_a}, (B): {token_b}")

    data = json.load(open(data_path, encoding='utf-8'))

    # Steering hook - adds vector to residual stream output
    def make_hook(vec, mult):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + mult * vec.to(device=hidden.device, dtype=hidden.dtype)
                return (hidden,) + output[1:]
            return output + mult * vec.to(device=output.device, dtype=output.dtype)
        return hook

    # Baseline (no steering)
    baseline_syco = 0
    print("Computing baseline...")
    for item in tqdm(data):
        logp_a, logp_b = get_ab_logprobs(
            model, tokenizer, item['input'] + '\n', token_a, token_b, device
        )
        pred = "A" if logp_a > logp_b else "B"
        gold = item['matching'].strip()[1]  # "(A)" -> "A"
        if pred == gold:
            baseline_syco += 1

    baseline_rate = baseline_syco / len(data)
    print(f"Baseline sycophancy rate: {baseline_syco}/{len(data)} = {baseline_rate*100:.1f}%")

    # Load and optionally normalize vector
    vector = torch.load(vector_path, map_location=device, weights_only=True).to(torch.bfloat16)
    if normalize:
        vector = vector / vector.norm()
    print(f"Vector norm: {vector.norm().item():.4f}")

    # Apply steering hook to target layer
    # Qwen3.5 uses model.model.layers for both self_attn and linear_attn layers
    target_layer = model.model.layers[layer]
    handle = target_layer.register_forward_hook(make_hook(vector, multiplier))

    # Steered evaluation
    steered_syco = 0
    print(f"Computing steered (layer={layer}, multiplier={multiplier})...")
    for item in tqdm(data):
        logp_a, logp_b = get_ab_logprobs(
            model, tokenizer, item['input'] + '\n', token_a, token_b, device
        )
        pred = "A" if logp_a > logp_b else "B"
        gold = item['matching'].strip()[1]
        if pred == gold:
            steered_syco += 1

    handle.remove()

    steered_rate = steered_syco / len(data)
    delta = (steered_rate - baseline_rate) * 100
    print(f"Steered sycophancy rate: {steered_syco}/{len(data)} = {steered_rate*100:.1f}%")
    print(f"Delta: {delta:+.1f}pp")

    result = {
        "model": model_name,
        "layer": layer,
        "multiplier": multiplier,
        "n_samples": len(data),
        "baseline_sycophancy_rate": round(baseline_rate * 100, 2),
        "steered_sycophancy_rate": round(steered_rate * 100, 2),
        "delta_pp": round(delta, 2),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(output_path, "w"), indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--vector", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--multiplier", type=float, default=-1.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    evaluate(args.data, args.vector, args.layer, args.multiplier, args.output, normalize=args.normalize)
