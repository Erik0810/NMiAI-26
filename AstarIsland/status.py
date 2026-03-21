"""Quick check of round status and budget."""
from client import AstarClient
c = AstarClient()
rounds = c.get_rounds()
for r in rounds:
    rn = r["round_number"]
    status = r["status"]
    closes = r.get("closes_at", "N/A")
    print(f"Round {rn}: {status} (closes: {closes})")
budget = c.get_budget()
print(f"\nBudget: {budget['queries_used']}/{budget['queries_max']}")
lb = c.get_leaderboard()
print(f"\nLeaderboard (top 5):")
for entry in lb[:5]:
    print(f"  {entry.get('rank', '?')}. {entry.get('team_name', 'N/A')}: {entry.get('best_score', 0):.1f}")
