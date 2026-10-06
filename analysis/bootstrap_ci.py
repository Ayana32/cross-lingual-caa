import json, sys, numpy as np
sys.path.insert(0, 'cross_lingual')
from parse_utils import parse_prediction

np.random.seed(42)
langs = ['en', 'it', 'zh', 'ko', 'kk']

results_map = {
    'Qwen3.5': {
        'en': 'qwen35_syco_en_m-1.5_layer15',
        'it': 'qwen35_syco_it_m-1.5_layer15',
        'zh': 'qwen35_syco_zh_m-1.5_layer15',
        'ko': 'qwen35_syco_ko_m-1.5_layer15',
        'kk': 'qwen35_syco_kk_m-1.5_layer15',
    },
    'Gemma4': {
        'en': 'gemma4_syco_en_m-1.0_layer26',
        'it': 'gemma4_syco_it_m-1.0_layer26',
        'zh': 'gemma4_syco_zh_m-1.0_layer26',
        'ko': 'gemma4_syco_ko_m-1.0_layer26',
        'kk': 'gemma4_syco_kk_m-1.0_layer26',
    },
    'OLMo3': {
        'en': 'olmo3_syco_en_m-1.5_layer14',
        'it': 'olmo3_syco_it_m-1.5_layer14',
        'zh': 'olmo3_syco_zh_m-1.5_layer14',
        'ko': 'olmo3_syco_ko_m-1.5_layer14',
        'kk': 'olmo3_syco_kk_m-1.5_layer14',
    },
}

print(f"{'Model':<10} {'Lang':<5} {'N':>5} {'Baseline':>10} {'Steered':>10} {'Delta':>8} {'95% CI':>18} {'Sig':>5}")
print('-' * 80)

for model, dirs in results_map.items():
    for lang in langs:
        try:
            source = json.load(open(f'data/sycophancy/{lang}/eval.json'))
            r = json.load(open(f'results/{dirs[lang]}/syco_{lang}_results.json'))

            # Use paired-parseable denominator (consistent with analyze_sweep.py)
            data_pairs = []
            for i, item in enumerate(r):
                gold = source[i]['matching']
                orig = item.get('orig_pred', '')
                orig = orig[0] if isinstance(orig, list) else orig
                pred = item.get('pred', '')
                pred = pred[0] if isinstance(pred, list) else pred
                p_orig = parse_prediction(orig)
                p_pred = parse_prediction(pred)
                if p_orig is None or p_pred is None:
                    continue
                data_pairs.append((p_orig, p_pred, gold))

            n = len(data_pairs)
            orig_correct = sum(1 for o, p, g in data_pairs if o == g)
            steer_correct = sum(1 for o, p, g in data_pairs if p == g)

            baseline = orig_correct / n * 100
            steered = steer_correct / n * 100
            delta = steered - baseline

            # Bootstrap CI on delta (5000 resamples)
            boot_deltas = []
            for _ in range(5000):
                sample = [data_pairs[i] for i in np.random.choice(n, size=n, replace=True)]
                b_orig = sum(1 for o, p, g in sample if o == g)
                b_steer = sum(1 for o, p, g in sample if p == g)
                boot_deltas.append((b_steer - b_orig) / n * 100)

            lo = np.percentile(boot_deltas, 2.5)
            hi = np.percentile(boot_deltas, 97.5)
            sig = 'yes' if lo > 0 or hi < 0 else 'no'

            print(f"{model:<10} {lang:<5} {n:>5} {baseline:>9.1f}% {steered:>9.1f}% {delta:>+7.1f}pp [{lo:+.1f}, {hi:+.1f}] {sig:>5}")
        except Exception as e:
            print(f"{model:<10} {lang:<5} ERROR: {e}")
    print()
