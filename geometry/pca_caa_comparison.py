"""
CAA vs PCA vector comparison analysis.
"""
import torch
import torch.nn.functional as F
import json
from pathlib import Path

BASE_EASYEDIT = '/users/acp25mk/EasyEdit/vectors'
BASE_CAA = '/users/acp25mk/cross-lingual-caa/vectors'
LAYERS = list(range(12, 27))

def load_vector(path):
    v = torch.load(path, map_location='cpu', weights_only=True).float()
    if v.ndim > 1:
        print(f"Warning: {path} has shape {tuple(v.shape)}, flattening.")
    v = v.flatten()
    return F.normalize(v, dim=0)

def cosine(a, b):
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sim, abs(sim)

paths = {
    'caa_syco': {
        'EN': f'{BASE_CAA}/Qwen2.5-7B-Instruct/sycophancy/caa_vector_multiple_choice',
        'KO': f'{BASE_CAA}/Qwen2.5-7B-Instruct-ko/sycophancy_ko/caa_vector_multiple_choice',
        'ES': f'{BASE_CAA}/Qwen2.5-7B-Instruct-es/sycophancy_es/caa_vector_multiple_choice',
        'KK': f'{BASE_CAA}/Qwen2.5-7B-Instruct-kk/sycophancy_kk/caa_vector_multiple_choice',
    },
    'pca_syco': {
        'EN': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct/pca_vector_multiple_choice',
        'KO': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-ko/pca_vector_multiple_choice',
        'ES': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-es/pca_vector_multiple_choice',
        'KK': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-kk/pca_vector_multiple_choice',
    },
    'caa_semeval': {
        'EN': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-en/semeval_en/caa_vector_multiple_choice',
        'KO': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-ko/semeval_ko/caa_vector_multiple_choice',
        'ES': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-es/semeval_es/caa_vector_multiple_choice',
        'KK': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-kk/semeval_kk/caa_vector_multiple_choice',
    },
    'pca_semeval': {
        'EN': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-en/pca_vector_multiple_choice',
        'KO': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-ko/pca_vector_multiple_choice',
        'ES': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-es/pca_vector_multiple_choice',
        'KK': f'{BASE_EASYEDIT}/Qwen2.5-7B-Instruct-semeval-kk/pca_vector_multiple_choice',
    },
}

results = {}
errors = []

# 1. CAA vs PCA (같은 언어, 같은 task)
print("\n=== 1. CAA vs PCA (same lang, same task) ===")
for task in ['syco', 'semeval']:
    results[f'caa_vs_pca_{task}'] = {}
    print(f"\n[{task}] cosine / abs(cosine)")
    print(f"{'Layer':<6}", end='')
    for lang in ['EN', 'KO', 'ES', 'KK']:
        print(f"{lang+' cos':>10}{lang+' abs':>10}", end='')
    print()
    print('-' * 86)
    for layer in LAYERS:
        print(f"{layer:<6}", end='')
        results[f'caa_vs_pca_{task}'][layer] = {}
        for lang in ['EN', 'KO', 'ES', 'KK']:
            try:
                caa = load_vector(f"{paths[f'caa_{task}'][lang]}/layer_{layer}.pt")
                pca = load_vector(f"{paths[f'pca_{task}'][lang]}/layer_{layer}.pt")
                sim, abs_sim = cosine(caa, pca)
                results[f'caa_vs_pca_{task}'][layer][lang] = {'cosine': sim, 'abs_cosine': abs_sim}
                print(f"{sim:>10.3f}{abs_sim:>10.3f}", end='')
            except Exception as e:
                err_msg = f"caa_vs_pca_{task} L{layer} {lang}: {e}"
                errors.append(err_msg)
                results[f'caa_vs_pca_{task}'][layer][lang] = None
                print(f"{'ERR':>10}{'ERR':>10}", end='')
        print()

# 2. Cross-lingual CAA sycophancy (EN vs others)
print("\n=== 2. Cross-lingual (CAA sycophancy, EN vs others) ===")
results['cross_lingual_caa_syco'] = {}
print(f"{'Layer':<6}", end='')
for lang in ['KO', 'ES', 'KK']:
    print(f"{'EN-'+lang+' cos':>12}{'EN-'+lang+' abs':>12}", end='')
print()
print('-' * 78)
for layer in LAYERS:
    print(f"{layer:<6}", end='')
    results['cross_lingual_caa_syco'][layer] = {}
    try:
        en = load_vector(f"{paths['caa_syco']['EN']}/layer_{layer}.pt")
        for lang in ['KO', 'ES', 'KK']:
            try:
                other = load_vector(f"{paths['caa_syco'][lang]}/layer_{layer}.pt")
                sim, abs_sim = cosine(en, other)
                results['cross_lingual_caa_syco'][layer][f'EN-{lang}'] = {'cosine': sim, 'abs_cosine': abs_sim}
                print(f"{sim:>12.3f}{abs_sim:>12.3f}", end='')
            except Exception as e:
                errors.append(f"cross_lingual L{layer} EN-{lang}: {e}")
                results['cross_lingual_caa_syco'][layer][f'EN-{lang}'] = None
                print(f"{'ERR':>12}{'ERR':>12}", end='')
    except Exception as e:
        errors.append(f"cross_lingual L{layer} EN load: {e}")
        for lang in ['KO', 'ES', 'KK']:
            results['cross_lingual_caa_syco'][layer][f'EN-{lang}'] = None
            print(f"{'ERR':>12}{'ERR':>12}", end='')
    print()

# 3. Cross-task CAA (syco vs semeval, 같은 언어)
print("\n=== 3. Cross-task (CAA syco vs semeval, same lang) ===")
results['cross_task_caa'] = {}
print(f"{'Layer':<6}", end='')
for lang in ['EN', 'KO', 'ES', 'KK']:
    print(f"{lang+' cos':>10}{lang+' abs':>10}", end='')
print()
print('-' * 86)
for layer in LAYERS:
    print(f"{layer:<6}", end='')
    results['cross_task_caa'][layer] = {}
    for lang in ['EN', 'KO', 'ES', 'KK']:
        try:
            syco = load_vector(f"{paths['caa_syco'][lang]}/layer_{layer}.pt")
            semeval = load_vector(f"{paths['caa_semeval'][lang]}/layer_{layer}.pt")
            sim, abs_sim = cosine(syco, semeval)
            results['cross_task_caa'][layer][lang] = {'cosine': sim, 'abs_cosine': abs_sim}
            print(f"{sim:>10.3f}{abs_sim:>10.3f}", end='')
        except Exception as e:
            errors.append(f"cross_task L{layer} {lang}: {e}")
            results['cross_task_caa'][layer][lang] = None
            print(f"{'ERR':>10}{'ERR':>10}", end='')
    print()

# 저장
output = '/users/acp25mk/cross-lingual-caa/results/pca_caa_comparison.json'
Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output}")

if errors:
    print(f"\n{len(errors)} errors:")
    for e in errors:
        print(f"  {e}")
else:
    print("\nNo errors!")
