import json
from client import AstarClient
c = AstarClient()

# Active round
ar = c.get_active_round()
if ar:
    rd = c.get_round_detail(ar["id"])
    budget = c.get_budget()
    print(f"ACTIVE: Round {rd.get('round_number','?')} | Map {rd['map_width']}x{rd['map_height']} | Seeds {rd['seeds_count']}")
    print(f"  Budget: {budget['queries_used']}/{budget['queries_max']} | Closes: {rd.get('closes_at','?')}")
else:
    print("No active round")

# My round scores
print("\nMY SCORES:")
my = c.get_my_rounds()
best_weighted = 0
for m in sorted(my, key=lambda x: x.get("round_number", 0)):
    s = m.get("round_score")
    w = m.get("round_weight", 0)
    weighted = (s or 0) * w
    marker = ""
    if weighted > best_weighted:
        best_weighted = weighted
        marker = " <-- BEST"
    print(f"  R{m.get('round_number'):2d}: score={str(s):>8s}  rank={str(m.get('rank','--')):>4s}/{str(m.get('total_teams','?')):>4s}  weight={w:.4f}  weighted={weighted:.2f}{marker}")

# Leaderboard top 10 + our position
lb = c.get_leaderboard()
print(f"\nLEADERBOARD (top 10 of {len(lb)}):")
for i, entry in enumerate(lb[:10]):
    print(f"  #{i+1}: {entry.get('team_name','?'):20s} score={entry.get('weighted_score', entry.get('score','?'))}")
for i, entry in enumerate(lb):
    vals = str(entry).lower()
    if "erik" in vals or "overby" in vals:
        print(f"\n>>> US: #{i+1} of {len(lb)} — {json.dumps(entry, default=str)[:300]}")
        break
