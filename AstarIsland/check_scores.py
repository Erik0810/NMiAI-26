import json
from client import AstarClient
c = AstarClient()

my = c.get_my_rounds()
print("Round scores and ranks:")
for m in sorted(my, key=lambda x: x.get("round_number", 0)):
    rn = m.get("round_number")
    sc = m.get("round_score")
    wt = m.get("round_weight")
    rank = m.get("rank")
    total = m.get("total_teams")
    submitted = m.get("seeds_submitted")
    status = m.get("status")
    seed_sc = m.get("seed_scores")
    print(f"  R{rn:2d}: score={str(sc):>8s}  weight={wt:.4f}  rank={str(rank):>4s}/{str(total):>4s}  submitted={submitted}  status={status}")

# Leaderboard
lb = c.get_leaderboard()
print(f"\nLeaderboard ({len(lb)} entries), top 10:")
for i, entry in enumerate(lb[:10]):
    name = entry.get("team_name", entry.get("username", entry.get("name", "?")))
    score = entry.get("total_score", entry.get("score", entry.get("weighted_score", "?")))
    print(f"  {i+1:2d}. {name}: {score}")

# Find us
for i, entry in enumerate(lb):
    vals = json.dumps(entry).lower()
    if "erik" in vals or "overby" in vals:
        name = entry.get("team_name", entry.get("username", entry.get("name", "?")))
        score = entry.get("total_score", entry.get("score", entry.get("weighted_score", "?")))
        print(f"\n>>> Us: #{i+1} {name}: {score}")
        print(f"    Full entry: {json.dumps(entry, default=str)[:300]}")
        break
