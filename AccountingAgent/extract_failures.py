"""Extract detailed logs for failing tasks in rev34."""
import json
import re

with open(r'd:\NMiAI\AccountingAgent\rev34_logs.json', encoding='utf-8-sig') as f:
    data = json.load(f)

data.reverse()

# Find task boundaries
tasks = []
current = None
for e in data:
    msg = e.get('textPayload', '')
    ts = e.get('timestamp', '')[:19]
    if '=== New task received ===' in msg:
        if current:
            tasks.append(current)
        current = {'start': ts, 'msgs': []}
    if current:
        current['msgs'].append((ts, msg))
    if 'Task completed' in msg and current:
        tasks.append(current)
        current = None

# Print full logs for tasks with errors or failures
# Focus on tasks 4, 5, 9, 10 which had errors
for task_idx in [3, 4, 8, 9]:  # 0-indexed: task 4, 5, 9, 10
    if task_idx >= len(tasks):
        continue
    t = tasks[task_idx]
    print(f"{'='*80}")
    print(f"TASK {task_idx+1}")
    print(f"{'='*80}")
    for ts, msg in t['msgs']:
        if any(kw in msg for kw in ['Prompt', 'Tool call:', 'Result:', 'error', 'Error', '_error', 'final message', 'Task completed', 'Agent turn', 'continuation']):
            print(f"{ts} | {msg[:400]}")
    print()
