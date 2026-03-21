"""
Extended LOO Validation: Compare 1D Linear, 1D Kernel, 2D IDW, 2D Kernel, and Blends.
Follows the same LOO approach as validate_smoothing.py but adds 2D methods.
"""
import json
import numpy as np
import os
from config import PROB_FLOOR

KNOWLEDGE_DIR = "knowledge_base"

# Known expansion rates per round (from prior analysis)
EXPANSION_RATES = {
    "1": 0.170, "2": 0.205, "3": 0.002, "4": 0.097, "5": 0.130,
    "6": 0.264, "7": 0.147, "8": 0.026, "9": 0.146, "10": 0.009,
    "11": 0.292, "12": 0.140, "13": 0.101, "14": 0.265,
}


def linear_interp(rates, dists, target):
    rates, dists = np.array(rates, float), np.array(dists, float)
    if len(rates) == 1: r = dists[0].copy()
    elif target <= rates[0]: r = dists[0].copy()
    elif target >= rates[-1]: r = dists[-1].copy()
    else:
        idx = max(0, min(int(np.searchsorted(rates, target)) - 1, len(rates) - 2))
        r0, r1 = rates[idx], rates[idx + 1]
        t = (target - r0) / max(r1 - r0, 1e-10)
        r = (1 - t) * dists[idx] + t * dists[idx + 1]
    r = np.maximum(r, PROB_FLOOR); return r / r.sum()


def kernel_interp(rates, dists, target, bw=0.07):
    rates, dists = np.array(rates, float), np.array(dists, float)
    if len(rates) == 1: return (np.maximum(dists[0], PROB_FLOOR) / np.maximum(dists[0], PROB_FLOOR).sum())
    w = np.exp(-0.5 * ((rates - target) / bw) ** 2)
    w = np.maximum(w, 1e-10); w /= w.sum()
    r = np.dot(w, dists)
    r = np.maximum(r, PROB_FLOOR); return r / r.sum()


def idw_2d(surv, exp, dists, ts, te):
    surv, exp, dists = np.array(surv, float), np.array(exp, float), np.array(dists, float)
    if len(surv) == 1: r = dists[0].copy()
    else:
        sr = max(surv.max() - surv.min(), 0.01)
        er = max(exp.max() - exp.min(), 0.001)
        sn = (surv - surv.min()) / sr
        en = (exp - exp.min()) / er
        tsn = (ts - surv.min()) / sr
        ten = (te - exp.min()) / er
        d = np.sqrt((sn - tsn)**2 + (en - ten)**2)
        w = (1.0 / (d + 0.01)) ** 2; w /= w.sum()
        r = np.dot(w, dists)
    r = np.maximum(r, PROB_FLOOR); return r / r.sum()


def kernel_2d(surv, exp, dists, ts, te, bw_s=0.07, bw_e=0.10):
    surv, exp, dists = np.array(surv, float), np.array(exp, float), np.array(dists, float)
    if len(surv) == 1: r = dists[0].copy()
    else:
        w = np.exp(-0.5 * ((surv - ts) / bw_s)**2) * np.exp(-0.5 * ((exp - te) / bw_e)**2)
        w = np.maximum(w, 1e-10); w /= w.sum()
        r = np.dot(w, dists)
    r = np.maximum(r, PROB_FLOOR); return r / r.sum()


def main():
    surv_kb = json.load(open(os.path.join(KNOWLEDGE_DIR, "survival_indexed_kb.json")))
    exp_kb = json.load(open(os.path.join(KNOWLEDGE_DIR, "expansion_indexed_kb.json")))
    
    round_rates = surv_kb["round_survival_rates"]  # round_num_str -> survival rate
    context_keys = surv_kb["context_keys"]
    exp_context_keys = exp_kb["context_keys"]
    
    all_rounds = sorted(round_rates.keys(), key=lambda x: int(x))
    print(f"KB: {len(context_keys)} keys, {len(all_rounds)} rounds")
    
    methods = ["1D_linear", "1D_kernel", "2D_IDW", "2D_kernel", 
               "blend_5_5_IDW", "blend_5_5_kern", "1D_surv_only"]
    
    results = {m: {} for m in methods}
    
    for leave_rn in all_rounds:
        leave_surv = float(round_rates[leave_rn])
        leave_exp = EXPANSION_RATES.get(leave_rn, 0.13)
        
        for m in methods:
            results[m][leave_rn] = {"wkl": 0.0, "ent": 0.0, "cells": 0}
        
        for key, entry in context_keys.items():
            rates = np.array(entry["rates"], float)
            dists = np.array(entry["dists"], float)
            
            # Find which index has this round's rate
            idx_match = np.argmin(np.abs(rates - leave_surv))
            gt_dist = dists[idx_match].copy()
            gt_dist = np.maximum(gt_dist, 1e-10)
            gt_dist /= gt_dist.sum()
            
            ent = -np.sum(gt_dist * np.log(gt_dist + 1e-10))
            if ent < 0.01:
                continue
            
            # LOO: remove this round
            mask = np.ones(len(rates), bool)
            mask[idx_match] = False
            loo_rates = rates[mask]
            loo_dists = dists[mask]
            if len(loo_rates) == 0:
                continue
            
            # Also build LOO 2D data for expansion KB
            loo_2d_surv, loo_2d_exp, loo_2d_dists = [], [], []
            if key in exp_context_keys:
                exp_entry = exp_context_keys[key]
                e_surv = np.array(exp_entry["survival_rates"], float)
                e_exp = np.array(exp_entry["expansion_rates"], float)
                e_dists = np.array(exp_entry["dists"], float)
                # Match by survival rate
                e_idx = np.argmin(np.abs(e_surv - leave_surv))
                e_mask = np.ones(len(e_surv), bool)
                e_mask[e_idx] = False
                loo_2d_surv = e_surv[e_mask]
                loo_2d_exp = e_exp[e_mask]
                loo_2d_dists = e_dists[e_mask]
            
            for method in methods:
                if method == "1D_linear":
                    pred = linear_interp(loo_rates, loo_dists, leave_surv)
                elif method == "1D_kernel":
                    pred = kernel_interp(loo_rates, loo_dists, leave_surv)
                elif method == "1D_surv_only":
                    # kernel with wider bandwidth
                    pred = kernel_interp(loo_rates, loo_dists, leave_surv, bw=0.10)
                elif method == "2D_IDW":
                    if len(loo_2d_surv) > 0:
                        pred = idw_2d(loo_2d_surv, loo_2d_exp, loo_2d_dists, leave_surv, leave_exp)
                    else:
                        pred = kernel_interp(loo_rates, loo_dists, leave_surv)
                elif method == "2D_kernel":
                    if len(loo_2d_surv) > 0:
                        pred = kernel_2d(loo_2d_surv, loo_2d_exp, loo_2d_dists, leave_surv, leave_exp)
                    else:
                        pred = kernel_interp(loo_rates, loo_dists, leave_surv)
                elif method == "blend_5_5_IDW":
                    p1 = kernel_interp(loo_rates, loo_dists, leave_surv)
                    if len(loo_2d_surv) > 0:
                        p2 = idw_2d(loo_2d_surv, loo_2d_exp, loo_2d_dists, leave_surv, leave_exp)
                    else:
                        p2 = p1
                    pred = 0.5 * p1 + 0.5 * p2
                    pred = np.maximum(pred, PROB_FLOOR); pred /= pred.sum()
                elif method == "blend_5_5_kern":
                    p1 = kernel_interp(loo_rates, loo_dists, leave_surv)
                    if len(loo_2d_surv) > 0:
                        p2 = kernel_2d(loo_2d_surv, loo_2d_exp, loo_2d_dists, leave_surv, leave_exp)
                    else:
                        p2 = p1
                    pred = 0.5 * p1 + 0.5 * p2
                    pred = np.maximum(pred, PROB_FLOOR); pred /= pred.sum()
                
                kl = np.sum(gt_dist * np.log(gt_dist / np.maximum(pred, 1e-10)))
                results[method][leave_rn]["wkl"] += ent * kl
                results[method][leave_rn]["ent"] += ent
                results[method][leave_rn]["cells"] += 1
    
    # Print results
    print(f"\n{'Method':<18s}", end="")
    for rn in all_rounds:
        print(f" R{rn:>2s}", end="")
    print("   AVG")
    print("-" * (18 + 5 * len(all_rounds) + 8))
    
    method_avgs = {}
    for method in methods:
        scores = []
        print(f"{method:<18s}", end="")
        for rn in all_rounds:
            r = results[method][rn]
            if r["ent"] > 0:
                wkl = r["wkl"] / r["ent"]
                sc = max(0, min(100, 100 * np.exp(-3 * wkl)))
                scores.append(sc)
                print(f" {sc:4.1f}", end="")
            else:
                print(f"   --", end="")
        avg = np.mean(scores) if scores else 0
        method_avgs[method] = avg
        print(f"  {avg:5.1f}")
    
    # Differences from 2D_IDW (current model)
    print(f"\nDiff from 2D_IDW (current model):")
    print(f"{'Method':<18s}", end="")
    for rn in all_rounds:
        print(f" R{rn:>2s}", end="")
    print("   AVG")
    print("-" * (18 + 5 * len(all_rounds) + 8))
    
    for method in methods:
        diffs = []
        print(f"{method:<18s}", end="")
        for rn in all_rounds:
            r = results[method][rn]
            r2 = results["2D_IDW"][rn]
            if r["ent"] > 0 and r2["ent"] > 0:
                s1 = 100 * np.exp(-3 * r["wkl"] / r["ent"])
                s2 = 100 * np.exp(-3 * r2["wkl"] / r2["ent"])
                d = s1 - s2
                diffs.append(d)
                print(f" {d:+4.1f}", end="")
            else:
                print(f"   --", end="")
        avg_d = np.mean(diffs) if diffs else 0
        print(f"  {avg_d:+5.2f}")
    
    # Best method per round
    print(f"\nBest method per round:")
    for rn in all_rounds:
        best_m = max(methods, key=lambda m: (
            100 * np.exp(-3 * results[m][rn]["wkl"] / results[m][rn]["ent"]) 
            if results[m][rn]["ent"] > 0 else 0
        ))
        r = results[best_m][rn]
        sc = 100 * np.exp(-3 * r["wkl"] / r["ent"]) if r["ent"] > 0 else 0
        print(f"  R{rn}: {best_m} ({sc:.1f})")


if __name__ == "__main__":
    main()
