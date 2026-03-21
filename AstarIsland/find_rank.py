from client import AstarClient
import json
c = AstarClient()
lb = c.get_leaderboard()

# Get our team_id from my-rounds
my = c.get_my_rounds()
my_team_id = my[0].get("team_id") if my else None

# Search by team_id or look for ourselves
for i, e in enumerate(lb):
    if my_team_id and e.get("team_id") == my_team_id:
        print(f"Found by team_id: #{i+1} - {e['team_name']}: {e['weighted_score']}")
        print(f"  rounds_participated: {e.get('rounds_participated')}")
        print(f"  hot_streak_score: {e.get('hot_streak_score')}")
        break

# Also look at what our team_id is from config
print(f"\nOur team_id from my-rounds: {my_team_id}")
print(f"Sample keys from my-rounds[0]: {[k for k in my[0].keys() if 'team' in k.lower() or 'user' in k.lower()]}")

# Print around rank 60-70
print("\nRanks 55-80:")
for i, e in enumerate(lb[54:80], start=55):
    print(f"  {i}. {e['team_name']}: {e['weighted_score']} (participated: {e.get('rounds_participated')})")
