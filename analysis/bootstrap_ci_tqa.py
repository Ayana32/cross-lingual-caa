import json, sys, numpy as np
sys.path.insert(0, 'cross_lingual')
from parse_utils import parse_prediction

np.random.seed(42)
langs = ['en', 'it', 'zh', 'ko', 'kk']

results_map = {
    'Qwen3.5': {
        'en': ('tqa_qwen35_en_m1.5_layer13', 'en'),
        'it': ('tqa_qwen35_it_m1.5_layer11', 'it'),
        'zh': ('tqa_qwen35_zh_m1.5_layer17', 'zh'),
        'ko': ('tqa_qwen35_ko_m1.5_layer16', 'ko'),
        'kk': ('tqa_qwen35_kk_m1.5_layer11', 'kk'),
    },
    'Gemma4': {
        'en': ('tqa_gemma4_en_m1.0_layer23', 'en'),
        'it': ('tqa_gemma4_it_m1.0_layer23', 'it'),
        'zh': ('tqa_gemma4_zh_m1.0_layer23', 'zh'),
        'ko': ('tqa_gemma4_ko_m1.0_layer23', 'ko'),
        'kk': ('tqa_gemma4_kk_m1.0_layer23', 'kk'),
    },
    'OLMo3': {
        'en': ('tqa_olmo3_en_m1.5_layer14', 'en'),
        'it': ('tqa_olmo3_it_m1.5_layer14', 'it'),
        'zh': ('tqa_olmo3_zh_m1.5_layer14', 'zh'),
        'ko': ('tqa_olmo3_ko_m1.5_layer14', 'ko'),
        'kk': ('tqa_olmo3_kk_m1.5_layer14', 'kk'),
    },
}

def bootstrap_ci(pairs, n_boot=1000):
    pairs = np.array(pairs)
    deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(len(pairs), len(pairs), replace=True)
        s = pairs[idx]
        deltas.append((s[:,1].mean() - s[:,0].mean()) * 100)
    return np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)

print(f"{'Model':<10} {'Lang':<5} {'N':>5} {'Baseline':>10} {'Steered':>10} {'Delta':>8} {'95% CI':>18} {'Sig':>5}")
print('-' * 80)

for model, dirs in results_map.items():
    for lang in langs:
        try:
            result_dir, data_lang = dirs[lang]
            source = json.load(open(f'data/truthfulqa/{data_lang}/eval.json'))
            r = json.load(open(f'results/{result_dir}/tqa_{data_lang}_results.json'))
        except Exception as e:
            print(f"{model:<10} {lang:<5} MISSING ({e})")
            continue

        pairs = []
        for res, src in zip(r, source):
            op = parse_prediction(res.get('orig_pred', [''])[0])
            sp = parse_prediction(res.get('pred', [''])[0])
            if op is None or sp is None: continue
            ref = src.get('matching', '')
            pairs.append([int(op == ref), int(sp == ref)])

        if not pairs:
            print(f"{model:<10} {lang:<5} NO PAIRED")
            continue

        pairs_arr = np.array(pairs)
        n = len(pairs_arr)
        baseline = pairs_arr[:,0].mean() * 100
        steered  = pairs_arr[:,1].mean() * 100
        delta    = steered - baseline
        lo, hi   = bootstrap_ci(pairs_arr)
        sig      = 'yes' if (lo > 0 or hi < 0) else 'no'
        ci_str   = f"[{lo:+.1f}, {hi:+.1f}]"
        print(f"{model:<10} {lang:<5} {n:>5} {baseline:>9.1f}% {steered:>9.1f}% {delta:>+7.1f}pp {ci_str:>18}   {sig:>3}")
    print()
