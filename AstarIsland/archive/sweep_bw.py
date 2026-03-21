"""Quick sweep of 2D kernel bandwidths for both survival and expansion axes."""
import json, numpy as np, os
from config import PROB_FLOOR

EXPANSION_RATES = {
    "1": 0.170, "2": 0.205, "3": 0.002, "4": 0.097, "5": 0.130,
    "6": 0.264, "7": 0.147, "8": 0.026, "9": 0.146, "10": 0.009,
    "11": 0.292, "12": 0.140, "13": 0.101, "14": 0.265,
}

def kernel_2d(surv, exp, dists, ts, te, bw_s, bw_e):
    surv, exp, dists = np.array(surv, float), np.array(exp, float), np.array(dists, float)
    if len(surv) == 1: return np.maximum(dists[0], PROB_FLOOR) / np.maximum(dists[0], PROB_FLOOR).sum()
    w = np.exp(-0.5*((surv-ts)/bw_s)**2) * np.exp(-0.5*((exp-te)/bw_e)**2)
    w = np.maximum(w, 1e-10); w /= w.sum()
    r = np.dot(w, dists); r = np.maximum(r, PROB_FLOOR); return r / r.sum()

def evaluate(surv_kb, exp_kb, bw_s, bw_e):
    round_rates = surv_kb["round_survival_rates"]
    all_rounds = sorted(round_rates.keys(), key=lambda x: int(x))
    total_wkl, total_ent = 0.0, 0.0
    
    for leave_rn in all_rounds:
        leave_surv = float(round_rates[leave_rn])
        leave_exp = EXPANSION_RATES.get(leave_rn, 0.13)
        
        for key, entry in surv_kb["context_keys"].items():
            rates = np.array(entry["rates"], float)
            dists = np.array(entry["dists"], float)
            idx_match = np.argmin(np.abs(rates - leave_surv))
            gt = np.maximum(dists[idx_match], 1e-10); gt /= gt.sum()
            ent = -np.sum(gt * np.log(gt + 1e-10))
            if ent < 0.01: continue
            
            if key not in exp_kb["context_keys"]: continue
            exp_entry = exp_kb["context_keys"][key]
            e_surv = np.array(exp_entry["survival_rates"], float)
            e_exp = np.array(exp_entry["expansion_rates"], float)
            e_dists = np.array(exp_entry["dists"], float)
            e_idx = np.argmin(np.abs(e_surv - leave_surv))
            mask = np.ones(len(e_surv), bool); mask[e_idx] = False
            
            pred = kernel_2d(e_surv[mask], e_exp[mask], e_dists[mask], leave_surv, leave_exp, bw_s, bw_e)
            kl = np.sum(gt * np.log(gt / np.maximum(pred, 1e-10)))
            total_wkl += ent * kl; total_ent += ent
    
    return 100 * np.exp(-3 * total_wkl / total_ent) if total_ent > 0 else 0

surv_kb = json.load(open("knowledge_base/survival_indexed_kb.json"))
exp_kb = json.load(open("knowledge_base/expansion_indexed_kb.json"))

print(f"{'bw_surv':>8s} {'bw_exp':>8s} {'Score':>8s}")
print("-" * 28)
best_score, best_params = 0, None
for bw_s in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]:
    for bw_e in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
        sc = evaluate(surv_kb, exp_kb, bw_s, bw_e)
        if sc > best_score:
            best_score = sc
            best_params = (bw_s, bw_e)
            print(f"{bw_s:8.2f} {bw_e:8.2f} {sc:8.2f}  ***")
        elif sc > best_score - 0.5:
            print(f"{bw_s:8.2f} {bw_e:8.2f} {sc:8.2f}")

print(f"\nBest: bw_surv={best_params[0]}, bw_exp={best_params[1]}, score={best_score:.2f}")
