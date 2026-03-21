"""Analyze where our model loses points vs oracle to find improvement opportunities."""
import numpy as np
import json
import os
from client import AstarClient, grid_to_class_map
from model_v3 import build_prediction_v3, load_survival_kb, estimate_survival_mle
from config import PROB_FLOOR, CLASS_NAMES

c = AstarClient()
kb = load_survival_kb()
rounds = c.get_my_rounds()

print("=" * 80)
print("GAP ANALYSIS: V3 model vs Oracle vs Perfect")
print("=" * 80)

results = []

for rd in sorted(rounds, key=lambda r: r['round_number']):
    rn = rd['round_number']
    if rd['status'] != 'completed':
        continue
    
    d = c.get_round_detail(rd['id'])
    W, H = d['map_width'], d['map_height']
    
    # Load observations if available
    obs_path = f'round_data/observations_{rd["id"]}.json'
    obs = []
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            obs = json.load(f)
    
    all_states = {si: d['initial_states'][si] for si in range(5)}
    
    # Get GT survival rate
    gt_survs = []
    for si in range(5):
        try:
            a = c.get_analysis(rd['id'], si)
            gt = np.array(a['ground_truth'])
            setts = d['initial_states'][si]['settlements']
            for s in setts:
                if s['alive']:
                    gt_survs.append(float(gt[s['y'], s['x'], 1] + gt[s['y'], s['x'], 2]))
        except:
            pass
    
    if not gt_survs:
        continue
    gt_surv = np.mean(gt_survs)
    
    # MLE estimate
    if obs:
        mle_s, _ = estimate_survival_mle(obs, all_states, W, H, kb)
    else:
        mle_s = 0.28
    
    seed_data = []
    for si in range(5):
        try:
            a = c.get_analysis(rd['id'], si)
        except:
            continue
        gt = np.array(a['ground_truth'])
        state = d['initial_states'][si]
        cm = grid_to_class_map(state['grid'])
        rg = np.array(state['grid'])
        
        # Prediction with MLE survival
        pred = build_prediction_v3(cm, rg, state['settlements'], obs, si, W, H,
            survival_kb=kb, estimated_survival=mle_s,
            all_seeds_settlements={i: all_states[i]['settlements'] for i in range(5)})
        
        # Prediction with oracle survival
        pred_o = build_prediction_v3(cm, rg, state['settlements'], obs, si, W, H,
            survival_kb=kb, estimated_survival=gt_surv,
            all_seeds_settlements={i: all_states[i]['settlements'] for i in range(5)})
        
        gt_s = np.maximum(gt, 1e-10)
        pred_s = np.maximum(pred, 1e-10)
        pred_os = np.maximum(pred_o, 1e-10)
        
        kl_mle = np.sum(gt_s * np.log(gt_s / pred_s), axis=-1)
        kl_oracle = np.sum(gt_s * np.log(gt_s / pred_os), axis=-1)
        ent = -np.sum(gt_s * np.log(gt_s), axis=-1)
        dyn = ent > 0.01
        
        wkl_mle = np.sum(ent[dyn] * kl_mle[dyn]) / np.sum(ent[dyn])
        wkl_oracle = np.sum(ent[dyn] * kl_oracle[dyn]) / np.sum(ent[dyn])
        sc_mle = 100 * np.exp(-3 * wkl_mle)
        sc_oracle = 100 * np.exp(-3 * wkl_oracle)
        
        # Per-terrain breakdown for oracle
        for terrain, raw_code in [('Plains', 11), ('Forest', 4)]:
            mask = (rg == raw_code) & dyn
            if np.any(mask):
                twkl = np.sum(ent[mask] * kl_oracle[mask]) / np.sum(ent[mask])
                seed_data.append({
                    'seed': si, 'sc_mle': sc_mle, 'sc_oracle': sc_oracle,
                    f'wkl_{terrain.lower()}': twkl
                })
            else:
                seed_data.append({
                    'seed': si, 'sc_mle': sc_mle, 'sc_oracle': sc_oracle,
                })
        
        # Find worst cells (oracle prediction)
        worst_idx = np.argsort(kl_oracle.ravel())[-5:]
        if si == 0:
            worst_cells = []
            for idx in worst_idx[::-1]:
                y, x = divmod(idx, W)
                raw = int(rg[y, x])
                gt_dist = gt[y, x]
                pred_dist = pred_o[y, x]
                gt_cls = CLASS_NAMES[np.argmax(gt_dist)]
                pred_cls = CLASS_NAMES[np.argmax(pred_dist)]
                worst_cells.append(f"({x},{y}) raw={raw} gt={gt_cls}({gt_dist[np.argmax(gt_dist)]:.2f}) pred={pred_cls}({pred_dist[np.argmax(pred_dist)]:.2f}) kl={kl_oracle[y,x]:.3f}")
    
    if not seed_data:
        continue
    
    avg_mle = np.mean([s['sc_mle'] for s in seed_data[:5]])
    avg_oracle = np.mean([s['sc_oracle'] for s in seed_data[:5]])
    has_obs = "YES" if obs else "no"
    submitted = rd.get('round_score', 0) or 0
    
    print(f"\nR{rn}: GT_surv={gt_surv:.3f} MLE={mle_s:.3f} | V3={avg_mle:.1f} oracle={avg_oracle:.1f} submitted={submitted:.1f} | obs={has_obs}")
    print(f"  Gap: MLE->oracle={avg_oracle-avg_mle:+.1f}  oracle->100={100-avg_oracle:+.1f}")
    if si == 0 and worst_cells:
        print(f"  Worst cells (seed 0 oracle):")
        for wc in worst_cells[:3]:
            print(f"    {wc}")
    
    results.append({
        'round': rn, 'gt_surv': gt_surv, 'mle': mle_s,
        'v3': avg_mle, 'oracle': avg_oracle, 'submitted': submitted,
        'has_obs': bool(obs),
    })

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Categorize gaps
surv_gap = []
kb_gap = []
for r in results:
    surv_gap.append(r['oracle'] - r['v3'])
    kb_gap.append(100 - r['oracle'])

print(f"\nAvg gap from survival estimation: {np.mean(surv_gap):.1f} pts")
print(f"Avg gap from KB quality (oracle->100): {np.mean(kb_gap):.1f} pts")
print(f"Avg gap from KB quality (w/ obs only): {np.mean([100-r['oracle'] for r in results if r['has_obs']]):.1f} pts")
print(f"Avg gap from KB quality (no obs): {np.mean([100-r['oracle'] for r in results if not r['has_obs']]):.1f} pts")

print(f"\nRounds with obs (survival est. gap):")
for r in results:
    if r['has_obs']:
        print(f"  R{r['round']}: surv_err={abs(r['gt_surv']-r['mle']):.3f} -> {r['oracle']-r['v3']:.1f}pts lost")

print(f"\nWorst oracle scores (KB quality ceiling):")
for r in sorted(results, key=lambda x: x['oracle']):
    print(f"  R{r['round']}: oracle={r['oracle']:.1f} surv={r['gt_surv']:.3f}")
