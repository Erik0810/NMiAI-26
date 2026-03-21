import json
kb = json.load(open('knowledge_base/survival_indexed_kb.json'))
keys = list(kb['context_keys'].keys())
print(f'{len(keys)} keys')

# Top-level info
print(f"Round survival rates: {kb.get('round_survival_rates', {})}")

# Sample key
for k in ['plains|near_sett_2|inland', 'forest|near_sett_2', 'settlement|inland|forest_2']:
    if k in kb['context_keys']:
        entry = kb['context_keys'][k]
        print(f"\nKey: {k}")
        print(f"  rates ({len(entry['rates'])}): {[f'{r:.3f}' for r in entry['rates']]}")
        print(f"  dists ({len(entry['dists'])}): first = {[f'{d:.4f}' for d in entry['dists'][0]]}")
        print(f"  last = {[f'{d:.4f}' for d in entry['dists'][-1]]}")
