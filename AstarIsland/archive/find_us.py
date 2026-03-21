from client import AstarClient
import json
c = AstarClient()
lb = c.get_leaderboard()

# Compute our expected weighted scores  
my_scores = {
    1: (27.32, 1.05),
    2: (80.06, 1.1025),
    6: (78.17, 1.3401),
    7: (64.22, 1.4071),
    10: (86.62, 1.6289),
    14: (77.43, 1.9799),
}

print("Our weighted round scores:")
best_weighted = 0
for rn, (sc, wt) in sorted(my_scores.items()):
    ws = sc * wt
    if ws > best_weighted:
        best_weighted = ws
    print(f"  R{rn:2d}: {sc:.2f} × {wt:.4f} = {ws:.2f}")

print(f"\nOur best weighted: {best_weighted:.2f}")

# Find us on the leaderboard by matching score
for i, e in enumerate(lb):
    ws = e.get("weighted_score", 0)
    if abs(ws - best_weighted) < 1.0:
        print(f"Likely us: #{i+1} {e['team_name']}: {ws}")

# Also look at our actual per-round ranks
my = c.get_my_rounds()
for m in sorted(my, key=lambda x: x.get("round_number", 0)):
    rn = m.get("round_number")
    sc = m.get("round_score")
    rank = m.get("rank")
    if sc is not None:
        print(f"  R{rn}: score={sc:.4f}, rank={rank}")
