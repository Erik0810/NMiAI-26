"""
LOO Validation: Compare 1D Linear, 1D Kernel, 2D IDW, and Blended approaches.
For each round, predict using KB built from the other 13 rounds, compare to actual.
"""
import numpy as np
import json

# Round info: {round_num: (survival_rate, expansion_rate)}
ROUND_PARAMS = {
    1: (0.4200, 0.170), 2: (0.4150, 0.205), 3: (0.2750, 0.002),
    4: (0.3300, 0.097), 5: (0.2350, 0.130), 6: (0.4150, 0.264),
    7: (0.4550, 0.147), 8: (0.2150, 0.026), 9: (0.1750, 0.146),
    10: (0.3300, 0.009), 11: (0.1200, 0.292), 12: (0.5000, 0.140),
    13: (0.1200, 0.101), 14: (0.4050, 0.265),
}

PROB_FLOOR = 0.01


def load_kb(path):
    with open(path) as f:
        return json.load(f)


def interpolate_linear(rates, dists, target):
    rates = np.array(rates, dtype=float)
    dists = np.array(dists, dtype=float)
    if len(rates) == 1:
        r = dists[0].copy()
    elif target <= rates[0]:
        r = dists[0].copy()
    elif target >= rates[-1]:
        r = dists[-1].copy()
    else:
        idx = int(np.searchsorted(rates, target)) - 1
        idx = max(0, min(idx, len(rates) - 2))
        r0, r1 = rates[idx], rates[idx + 1]
        if r1 - r0 < 1e-10:
            r = dists[idx].copy()
        else:
            t = (target - r0) / (r1 - r0)
            r = (1 - t) * dists[idx] + t * dists[idx + 1]
    r = np.maximum(r, PROB_FLOOR)
    return r / r.sum()


def interpolate_kernel(rates, dists, target, bw=0.07):
    rates = np.array(rates, dtype=float)
    dists = np.array(dists, dtype=float)
    if len(rates) == 1:
        r = dists[0].copy()
    else:
        diff = rates - target
        weights = np.exp(-0.5 * (diff / bw) ** 2)
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
        r = np.dot(weights, dists)
    r = np.maximum(r, PROB_FLOOR)
    return r / r.sum()


def interpolate_2d_idw(surv_rates, exp_rates, dists, target_surv, target_exp):
    surv = np.array(surv_rates, dtype=float)
    exp = np.array(exp_rates, dtype=float)
    all_dists = np.array(dists, dtype=float)
    if len(surv) == 1:
        r = all_dists[0].copy()
    else:
        surv_range = max(surv.max() - surv.min(), 0.01)
        exp_range = max(exp.max() - exp.min(), 0.001)
        surv_norm = (surv - surv.min()) / surv_range
        exp_norm = (exp - exp.min()) / exp_range
        ts = (target_surv - surv.min()) / surv_range
        te = (target_exp - exp.min()) / exp_range
        distances = np.sqrt((surv_norm - ts)**2 + (exp_norm - te)**2)
        eps = 0.01
        weights = 1.0 / (distances + eps)
        weights = weights ** 2
        weights /= weights.sum()
        r = np.dot(weights, all_dists)
    r = np.maximum(r, PROB_FLOOR)
    return r / r.sum()


def interpolate_2d_kernel(surv_rates, exp_rates, dists, target_surv, target_exp, bw_surv=0.07, bw_exp=0.10):
    """Kernel smoothing in 2D with separate bandwidths."""
    surv = np.array(surv_rates, dtype=float)
    exp = np.array(exp_rates, dtype=float)
    all_dists = np.array(dists, dtype=float)
    if len(surv) == 1:
        r = all_dists[0].copy()
    else:
        w_surv = np.exp(-0.5 * ((surv - target_surv) / bw_surv) ** 2)
        w_exp = np.exp(-0.5 * ((exp - target_exp) / bw_exp) ** 2)
        weights = w_surv * w_exp
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
        r = np.dot(weights, all_dists)
    r = np.maximum(r, PROB_FLOOR)
    return r / r.sum()


def kl_divergence(p, q):
    """KL(p || q) with floor protection."""
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    return np.sum(p * np.log(p / q))


def entropy(p):
    p = np.maximum(p, 1e-10)
    return -np.sum(p * np.log(p))


def score_from_kl(weighted_kl):
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def evaluate_round(held_out_round, surv_kb, exp_kb, methods):
    """
    Evaluate prediction quality for a held-out round.
    Uses distributions from the held-out round as ground truth.
    """
    rn = str(held_out_round)
    target_surv, target_exp = ROUND_PARAMS[held_out_round]
    
    surv_keys = surv_kb.get("context_keys", {})
    exp_keys = exp_kb.get("context_keys", {})
    
    results = {m: {"total_ent_kl": 0.0, "total_ent": 0.0} for m in methods}
    n_cells = 0
    
    for key in surv_keys:
        entry = surv_keys[key]
        if rn not in entry.get("round_data", {}):
            continue
        
        # Ground truth: the actual distribution for this round
        truth = np.array(entry["round_data"][rn], dtype=float)
        truth = np.maximum(truth, 1e-10)
        truth /= truth.sum()
        ent = entropy(truth)
        
        if ent < 0.01:
            continue
        
        # Build LOO versions: remove this round's data
        loo_rates = []
        loo_dists = []
        for rate, dist, rd_num in zip(entry["rates"], entry["dists"], entry.get("round_numbers", [])):
            if str(rd_num) != rn:
                loo_rates.append(rate)
                loo_dists.append(dist)
        
        if not loo_rates:
            continue
        
        # Also build LOO 2D
        if key in exp_keys:
            exp_entry = exp_keys[key]
            loo_2d_surv = []
            loo_2d_exp = []
            loo_2d_dists = []
            for s, e, d, rd in zip(
                exp_entry["survival_rates"], exp_entry["expansion_rates"],
                exp_entry["dists"], exp_entry.get("round_numbers", [])
            ):
                if str(rd) != rn:
                    loo_2d_surv.append(s)
                    loo_2d_exp.append(e)
                    loo_2d_dists.append(d)
        
        for method_name in methods:
            if method_name == "1D_linear":
                pred = interpolate_linear(loo_rates, loo_dists, target_surv)
            elif method_name == "1D_kernel":
                pred = interpolate_kernel(loo_rates, loo_dists, target_surv)
            elif method_name == "2D_IDW":
                if key in exp_keys and loo_2d_surv:
                    pred = interpolate_2d_idw(loo_2d_surv, loo_2d_exp, loo_2d_dists, target_surv, target_exp)
                else:
                    pred = interpolate_kernel(loo_rates, loo_dists, target_surv)
            elif method_name == "2D_kernel":
                if key in exp_keys and loo_2d_surv:
                    pred = interpolate_2d_kernel(loo_2d_surv, loo_2d_exp, loo_2d_dists, target_surv, target_exp)
                else:
                    pred = interpolate_kernel(loo_rates, loo_dists, target_surv)
            elif method_name == "blend_6_4":
                p1d = interpolate_kernel(loo_rates, loo_dists, target_surv)
                if key in exp_keys and loo_2d_surv:
                    p2d = interpolate_2d_idw(loo_2d_surv, loo_2d_exp, loo_2d_dists, target_surv, target_exp)
                else:
                    p2d = p1d
                pred = 0.6 * p1d + 0.4 * p2d
                pred = np.maximum(pred, PROB_FLOOR)
                pred /= pred.sum()
            elif method_name == "blend_5_5":
                p1d = interpolate_kernel(loo_rates, loo_dists, target_surv)
                if key in exp_keys and loo_2d_surv:
                    p2d = interpolate_2d_idw(loo_2d_surv, loo_2d_exp, loo_2d_dists, target_surv, target_exp)
                else:
                    p2d = p1d
                pred = 0.5 * p1d + 0.5 * p2d
                pred = np.maximum(pred, PROB_FLOOR)
                pred /= pred.sum()
            elif method_name == "blend_2d_kern":
                p1d = interpolate_kernel(loo_rates, loo_dists, target_surv)
                if key in exp_keys and loo_2d_surv:
                    p2d = interpolate_2d_kernel(loo_2d_surv, loo_2d_exp, loo_2d_dists, target_surv, target_exp)
                else:
                    p2d = p1d
                pred = 0.5 * p1d + 0.5 * p2d
                pred = np.maximum(pred, PROB_FLOOR)
                pred /= pred.sum()
            
            kl = kl_divergence(truth, pred)
            results[method_name]["total_ent_kl"] += ent * kl
            results[method_name]["total_ent"] += ent
        
        n_cells += 1
    
    scores = {}
    for method_name in methods:
        r = results[method_name]
        if r["total_ent"] > 0:
            weighted_kl = r["total_ent_kl"] / r["total_ent"]
            scores[method_name] = score_from_kl(weighted_kl)
        else:
            scores[method_name] = 0.0
    
    return scores, n_cells


def main():
    surv_kb = load_kb("knowledge_base/survival_indexed_kb.json")
    exp_kb = load_kb("knowledge_base/expansion_indexed_kb.json")
    
    # Check round_numbers presence
    sample_key = list(surv_kb["context_keys"].keys())[0]
    entry = surv_kb["context_keys"][sample_key]
    if "round_numbers" not in entry:
        # Reconstruct round_numbers from the round_data keys
        print("Reconstructing round_numbers from round_data...")
        for key in surv_kb["context_keys"]:
            e = surv_kb["context_keys"][key]
            if "round_data" in e:
                rn_list = sorted(e["round_data"].keys(), key=int)
                e["round_numbers"] = [int(r) for r in rn_list]
    
    # Same for expansion KB
    for key in exp_kb["context_keys"]:
        e = exp_kb["context_keys"][key]
        if "round_numbers" not in e:
            if "round_data" in e:
                rn_list = sorted(e["round_data"].keys(), key=int)
                e["round_numbers"] = [int(r) for r in rn_list]
    
    methods = ["1D_linear", "1D_kernel", "2D_IDW", "2D_kernel", "blend_5_5", "blend_6_4", "blend_2d_kern"]
    
    all_scores = {m: [] for m in methods}
    
    print(f"{'Round':>5s} {'Surv':>6s} {'Exp':>6s} " + 
          " ".join(f"{m:>12s}" for m in methods))
    print("-" * (22 + 13 * len(methods)))
    
    for rn in sorted(ROUND_PARAMS.keys()):
        surv, exp = ROUND_PARAMS[rn]
        scores, n_cells = evaluate_round(rn, surv_kb, exp_kb, methods)
        
        all_scores_str = " ".join(f"{scores.get(m, 0):>12.2f}" for m in methods)
        print(f"R{rn:>3d} {surv:>6.3f} {exp:>6.3f} {all_scores_str}  ({n_cells} keys)")
        
        for m in methods:
            all_scores[m].append(scores.get(m, 0))
    
    print("-" * (22 + 13 * len(methods)))
    avg_str = " ".join(f"{np.mean(all_scores[m]):>12.2f}" for m in methods)
    print(f"{'AVG':>5s} {'':>6s} {'':>6s} {avg_str}")
    
    # Show improvements over 2D_IDW baseline
    print(f"\nImprovement over 2D_IDW:")
    baseline = np.mean(all_scores["2D_IDW"])
    for m in methods:
        avg = np.mean(all_scores[m])
        diff = avg - baseline
        print(f"  {m:>15s}: {avg:.2f} ({diff:+.2f})")
    
    # Best method per round
    print(f"\nBest method per round:")
    for i, rn in enumerate(sorted(ROUND_PARAMS.keys())):
        best_m = max(methods, key=lambda m: all_scores[m][i])
        best_s = all_scores[best_m][i]
        worst_m = min(methods, key=lambda m: all_scores[m][i])
        worst_s = all_scores[worst_m][i]
        print(f"  R{rn}: best={best_m} ({best_s:.2f}), worst={worst_m} ({worst_s:.2f}), gap={best_s-worst_s:.2f}")


if __name__ == "__main__":
    main()
