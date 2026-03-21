"""
Deep cross-round analysis: learn from ALL completed rounds' ground truth.
Extracts detailed patterns, identifies systematic errors, and builds
improved priors for the model.
"""

import numpy as np
import json
import os
from collections import defaultdict
from client import AstarClient, grid_to_class_map
from config import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES, PROB_FLOOR
from learn import extract_cell_features, KNOWLEDGE_DIR


def analyze_all_rounds():
    """Download GT from all completed rounds, do deep cross-round analysis."""
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    
    completed = [r for r in my_rounds if r["status"] == "completed"]
    print(f"Found {len(completed)} completed rounds")
    for r in completed:
        print(f"  Round {r['round_number']}: score={r.get('round_score', 'N/A')}")
    
    # Collect all GT data across all rounds
    all_features = []
    all_gt_dists = []
    all_pred_dists = []
    all_meta = []  # (round_number, seed_idx, x, y)
    
    round_scores = {}
    
    for r in completed:
        round_id = r["id"]
        rn = r["round_number"]
        detail = client.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        seeds_count = detail["seeds_count"]
        
        print(f"\n{'='*60}")
        print(f"ROUND {rn} (id={round_id[:8]}...)")
        print(f"{'='*60}")
        
        seed_scores = []
        
        for seed_idx in range(seeds_count):
            try:
                analysis = client.get_analysis(round_id, seed_idx)
            except Exception as e:
                print(f"  Seed {seed_idx}: FAILED ({e})")
                continue
            
            gt = np.array(analysis["ground_truth"])
            pred_raw = analysis.get("prediction")
            if pred_raw is not None:
                pred = np.array(pred_raw)
            else:
                # No prediction submitted — use uniform as placeholder
                pred = np.full_like(gt, 1.0 / NUM_CLASSES)
            score = analysis.get("score", 0) or 0
            seed_scores.append(score)
            
            initial_grid = np.array(
                analysis.get("initial_grid") or detail["initial_states"][seed_idx]["grid"]
            )
            class_map = grid_to_class_map(initial_grid.tolist())
            settlements = detail["initial_states"][seed_idx]["settlements"]
            settlement_set = {(s["x"], s["y"]): s for s in settlements if s["alive"]}
            port_set = {(s["x"], s["y"]) for s in settlements if s["alive"] and s["has_port"]}
            
            # Per-cell analysis
            kl = np.sum(gt * np.log((gt + 1e-10) / (pred + 1e-10)), axis=-1)
            entropy = -np.sum(gt * np.log(gt + 1e-10), axis=-1)
            dynamic_mask = entropy > 0.01
            
            weighted_kl = np.sum(entropy[dynamic_mask] * kl[dynamic_mask]) / np.sum(entropy[dynamic_mask]) if np.any(dynamic_mask) else 0
            
            n_alive_settlements = len([s for s in settlements if s["alive"]])
            gt_settlement_cells = np.sum(np.argmax(gt, axis=-1) == 1)
            gt_port_cells = np.sum(np.argmax(gt, axis=-1) == 2)
            
            # Compute survival rate from GT
            survival_count = 0
            for (sx, sy), s in settlement_set.items():
                if gt[sy, sx, 1] + gt[sy, sx, 2] > 0.3:
                    survival_count += 1
            survival_rate = survival_count / max(1, len(settlement_set))
            
            print(f"  Seed {seed_idx}: score={score:.2f} wkl={weighted_kl:.4f} "
                  f"dynamic={np.sum(dynamic_mask)} "
                  f"sett_init={n_alive_settlements} sett_gt={gt_settlement_cells} port_gt={gt_port_cells} "
                  f"survival={survival_rate:.2f}")
            
            # Save GT data for offline validation
            gt_path = f"round_data/gt_r{rn}_seed{seed_idx}.npz"
            os.makedirs("round_data", exist_ok=True)
            np.savez_compressed(gt_path, 
                ground_truth=gt, prediction=pred, 
                initial_grid=initial_grid, class_map=class_map,
                round_number=rn, seed_idx=seed_idx)
            
            # Extract features for every cell
            for y in range(H):
                for x in range(W):
                    features = extract_cell_features(
                        x, y, class_map, initial_grid, settlement_set, port_set, W, H
                    )
                    # Add round-specific features
                    features["round_number"] = rn
                    features["seed_idx"] = seed_idx
                    features["kl"] = float(kl[y, x])
                    features["entropy"] = float(entropy[y, x])
                    features["x"] = x
                    features["y"] = y
                    
                    all_features.append(features)
                    all_gt_dists.append(gt[y, x].tolist())
                    all_pred_dists.append(pred[y, x].tolist())
                    all_meta.append((rn, seed_idx, x, y))
        
        if seed_scores:
            round_scores[rn] = {
                "avg": np.mean(seed_scores),
                "seeds": seed_scores,
            }
    
    print(f"\n{'='*60}")
    print("CROSS-ROUND ANALYSIS")
    print(f"{'='*60}")
    print(f"Total cells analyzed: {len(all_features)}")
    
    # ── 1. Error Analysis: Where are we losing the most points? ──
    print(f"\n--- ERROR BREAKDOWN BY INITIAL TERRAIN TYPE ---")
    error_by_type = defaultdict(lambda: {"kl_sum": 0, "entropy_sum": 0, "count": 0, "kls": []})
    
    for feat, gt, pred in zip(all_features, all_gt_dists, all_pred_dists):
        if feat["entropy"] < 0.01:
            continue
        
        gt_arr = np.array(gt)
        pred_arr = np.array(pred)
        kl = float(np.sum(gt_arr * np.log((gt_arr + 1e-10) / (pred_arr + 1e-10))))
        
        # Categorize
        if feat["is_settlement"] or feat["is_port"]:
            cat = "port" if feat["is_port"] else "settlement"
        elif feat["raw"] == 10:
            cat = "ocean"
        elif feat["raw"] == 11:
            cat = "plains"
        elif feat["class"] == 4:
            cat = "forest"
        elif feat["class"] == 5:
            cat = "mountain"
        else:
            cat = f"class_{feat['class']}"
        
        error_by_type[cat]["kl_sum"] += kl * feat["entropy"]
        error_by_type[cat]["entropy_sum"] += feat["entropy"]
        error_by_type[cat]["count"] += 1
        error_by_type[cat]["kls"].append(kl)
    
    total_weighted_kl = sum(v["kl_sum"] for v in error_by_type.values())
    for cat, data in sorted(error_by_type.items(), key=lambda x: -x[1]["kl_sum"]):
        avg_kl = data["kl_sum"] / max(data["entropy_sum"], 1e-10)
        pct = 100 * data["kl_sum"] / max(total_weighted_kl, 1e-10)
        p90 = np.percentile(data["kls"], 90) if data["kls"] else 0
        p99 = np.percentile(data["kls"], 99) if data["kls"] else 0
        print(f"  {cat:15s}: count={data['count']:5d} weighted_kl={data['kl_sum']:.3f} "
              f"avg_kl={avg_kl:.4f} p90={p90:.4f} p99={p99:.4f} "
              f"({pct:.1f}% of total error)")
    
    # ── 2. Learn improved distributions per context key ──
    print(f"\n--- IMPROVED DISTRIBUTIONS (cross-round average) ---")
    groups = defaultdict(lambda: {"dists": [], "preds": [], "kls": []})
    
    for feat, gt, pred in zip(all_features, all_gt_dists, all_pred_dists):
        cls = feat["class"]
        raw = feat["raw"]
        gt_arr = np.array(gt)
        pred_arr = np.array(pred)
        
        if cls == 5 or raw == 10:
            continue
        
        kl = float(np.sum(gt_arr * np.log((gt_arr + 1e-10) / (pred_arr + 1e-10))))
        
        # Use same key scheme as learn.py but with more granularity
        if feat["is_settlement"] or feat["is_port"]:
            stype = "port" if feat["is_port"] else "settlement"
            coast = "coastal" if feat["is_coastal"] else "inland"
            key = f"{stype}|{coast}|forest_{min(feat['n_forest'], 3)}"
            # Also make a more specific key
            key2 = f"{stype}|{coast}|forest_{min(feat['n_forest'], 3)}|sett_neighbors_{min(feat['n_settlement_neighbors'], 3)}"
            groups[key2]["dists"].append(gt_arr)
            groups[key2]["preds"].append(pred_arr)
            groups[key2]["kls"].append(kl)
        elif cls == 4:
            key = f"forest|near_sett_{min(feat['dist_to_nearest_settlement'], 5)}"
            # Add coastal distinction for forest
            if feat["is_coastal"]:
                key_c = f"forest|near_sett_{min(feat['dist_to_nearest_settlement'], 5)}|coastal"
                groups[key_c]["dists"].append(gt_arr)
                groups[key_c]["preds"].append(pred_arr)
                groups[key_c]["kls"].append(kl)
        elif raw == 11:
            coast = "coastal" if feat["is_coastal"] else "inland"
            key = f"plains|near_sett_{min(feat['dist_to_nearest_settlement'], 5)}|{coast}"
        else:
            key = f"class_{cls}|raw_{raw}"
        
        groups[key]["dists"].append(gt_arr)
        groups[key]["preds"].append(pred_arr)
        groups[key]["kls"].append(kl)
    
    # Build new KB from ALL rounds
    new_kb = {}
    for key, data in sorted(groups.items(), key=lambda x: -len(x[1]["dists"])):
        dists = np.array(data["dists"])
        avg = np.mean(dists, axis=0)
        std = np.std(dists, axis=0)
        avg_kl = np.mean(data["kls"])
        n = len(data["dists"])
        
        new_kb[key] = avg.tolist()
        
        if n >= 10:
            entropy = -np.sum(avg * np.log(avg + 1e-10))
            if entropy > 0.1:
                print(f"  {key} (n={n})")
                print(f"    GT:  {' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.3f}' for i in range(6))}")
                print(f"    std: {' '.join(f'{std[i]:.3f}' for i in range(6))}")
                print(f"    avg_kl={avg_kl:.4f}")
    
    # ── 3. Per-round variation analysis (how different are rounds?) ──
    print(f"\n--- PER-ROUND VARIATION (same key, different rounds) ---")
    round_groups = defaultdict(lambda: defaultdict(list))
    
    for feat, gt in zip(all_features, all_gt_dists):
        cls = feat["class"]
        raw = feat["raw"]
        rn = feat["round_number"]
        
        if cls == 5 or raw == 10:
            continue
        
        if feat["is_settlement"]:
            key = "settlement_all"
        elif feat["is_port"]:
            key = "port_all"
        elif cls == 4:
            d = min(feat["dist_to_nearest_settlement"], 5)
            key = f"forest_dist{d}"
        elif raw == 11:
            d = min(feat["dist_to_nearest_settlement"], 5)
            key = f"plains_dist{d}"
        else:
            continue
        
        round_groups[key][rn].append(np.array(gt))
    
    for key in sorted(round_groups.keys()):
        rd = round_groups[key]
        if len(rd) < 2:
            continue
        print(f"\n  {key}:")
        for rn in sorted(rd.keys()):
            avg = np.mean(rd[rn], axis=0)
            n = len(rd[rn])
            print(f"    Round {rn} (n={n:4d}): {' '.join(f'{CLASS_NAMES[i]}:{avg[i]:.3f}' for i in range(6))}")
    
    # ── 4. Biggest individual errors (worst predictions) ──
    print(f"\n--- TOP 50 WORST PREDICTIONS (by entropy-weighted KL) ---")
    errors = []
    for i, (feat, gt, pred) in enumerate(zip(all_features, all_gt_dists, all_pred_dists)):
        if feat["entropy"] < 0.01:
            continue
        gt_arr = np.array(gt)
        pred_arr = np.array(pred)
        kl = float(np.sum(gt_arr * np.log((gt_arr + 1e-10) / (pred_arr + 1e-10))))
        weighted = kl * feat["entropy"]
        errors.append((weighted, kl, feat, gt, pred))
    
    errors.sort(key=lambda x: x[0], reverse=True)
    for rank, (wkl, kl, feat, gt, pred) in enumerate(errors[:50]):
        gt_arr = np.array(gt)
        pred_arr = np.array(pred)
        gt_argmax = CLASS_NAMES[np.argmax(gt_arr)]
        pred_argmax = CLASS_NAMES[np.argmax(pred_arr)]
        
        if feat["is_settlement"]:
            ctype = "SETT"
        elif feat["is_port"]:
            ctype = "PORT"
        elif feat["raw"] == 11:
            ctype = "PLNS"
        elif feat["class"] == 4:
            ctype = "FRST"
        else:
            ctype = f"C{feat['class']}"
        
        print(f"  #{rank+1:2d} R{feat['round_number']}s{feat['seed_idx']} ({feat['x']},{feat['y']}) "
              f"{ctype} wkl={wkl:.4f} kl={kl:.4f}")
        print(f"       GT:   {' '.join(f'{gt_arr[i]:.3f}' for i in range(6))} ({gt_argmax})")
        print(f"       PRED: {' '.join(f'{pred_arr[i]:.3f}' for i in range(6))} ({pred_argmax})")
    
    # ── 5. What would a perfect KB score? ──
    print(f"\n--- THEORETICAL BEST WITH PERFECT KB ---")
    # If we used the cross-round average for each cell's context key, what score?
    for rn_target in sorted(round_scores.keys()):
        total_wkl_old = 0
        total_wkl_new = 0
        total_entropy = 0
        n_cells = 0
        
        for feat, gt, pred in zip(all_features, all_gt_dists, all_pred_dists):
            if feat["round_number"] != rn_target:
                continue
            if feat["entropy"] < 0.01:
                continue
            
            gt_arr = np.array(gt)
            pred_arr = np.array(pred)
            
            # Old KL
            kl_old = float(np.sum(gt_arr * np.log((gt_arr + 1e-10) / (pred_arr + 1e-10))))
            
            # New KL using cross-round KB (leave-one-out would be better but this is approximate)
            cls = feat["class"]
            raw = feat["raw"]
            if feat["is_settlement"] or feat["is_port"]:
                stype = "port" if feat["is_port"] else "settlement"
                coast = "coastal" if feat["is_coastal"] else "inland"
                key = f"{stype}|{coast}|forest_{min(feat['n_forest'], 3)}"
            elif cls == 4:
                key = f"forest|near_sett_{min(feat['dist_to_nearest_settlement'], 5)}"
            elif raw == 11:
                coast = "coastal" if feat["is_coastal"] else "inland"
                key = f"plains|near_sett_{min(feat['dist_to_nearest_settlement'], 5)}|{coast}"
            else:
                key = f"class_{cls}|raw_{raw}"
            
            if key in new_kb:
                new_pred = np.array(new_kb[key])
                new_pred = np.maximum(new_pred, PROB_FLOOR)
                new_pred = new_pred / new_pred.sum()
            else:
                new_pred = pred_arr  # fallback to old
            
            kl_new = float(np.sum(gt_arr * np.log((gt_arr + 1e-10) / (new_pred + 1e-10))))
            
            total_wkl_old += kl_old * feat["entropy"]
            total_wkl_new += kl_new * feat["entropy"]
            total_entropy += feat["entropy"]
            n_cells += 1
        
        if total_entropy > 0:
            wkl_old = total_wkl_old / total_entropy
            wkl_new = total_wkl_new / total_entropy
            score_old = 100 * np.exp(-3 * wkl_old)
            score_new = 100 * np.exp(-3 * wkl_new)
            print(f"  Round {rn_target}: old={score_old:.2f} -> new_kb={score_new:.2f} "
                  f"(wkl: {wkl_old:.4f} -> {wkl_new:.4f}) [{n_cells} dynamic cells]")
    
    # ── 6. Save the improved KB ──
    kb_path = os.path.join(KNOWLEDGE_DIR, "kb_all_rounds.json")
    with open(kb_path, "w") as f:
        json.dump(new_kb, f, indent=2)
    print(f"\nSaved improved KB ({len(new_kb)} keys) to {kb_path}")
    
    # Also save a version with count info for debugging
    kb_debug = {}
    for key, data in groups.items():
        avg = np.mean(data["dists"], axis=0)
        std = np.std(data["dists"], axis=0)
        kb_debug[key] = {
            "avg": avg.tolist(),
            "std": std.tolist(),
            "count": len(data["dists"]),
            "avg_kl": float(np.mean(data["kls"])),
        }
    
    kb_debug_path = os.path.join(KNOWLEDGE_DIR, "kb_debug.json")
    with open(kb_debug_path, "w") as f:
        json.dump(kb_debug, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ROUND SCORES SUMMARY")
    print(f"{'='*60}")
    for rn, data in sorted(round_scores.items()):
        print(f"  Round {rn}: avg={data['avg']:.2f} seeds={[f'{s:.1f}' for s in data['seeds']]}")


if __name__ == "__main__":
    analyze_all_rounds()
