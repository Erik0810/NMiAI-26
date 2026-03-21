import json

with open(r'd:\NMiAI\AccountingAgent\rev33_latest.json', encoding='utf-8-sig') as f:
    data = json.load(f)

data.reverse()

# Find task boundaries
tasks = []
current_task = None
for e in data:
    msg = e.get('textPayload', '')
    ts = e.get('timestamp', '')
    if '=== New task received ===' in msg:
        if current_task:
            tasks.append(current_task)
        current_task = {'start': ts, 'msgs': []}
    if current_task:
        current_task['msgs'].append((ts[:19], msg))
    if 'Task completed' in msg and current_task:
        current_task['end'] = ts
        tasks.append(current_task)
        current_task = None

print(f"Found {len(tasks)} tasks")
for i, t in enumerate(tasks):
    prompt_line = ''
    final_msg = ''
    api_count = 0
    errors = 0
    for ts, m in t['msgs']:
        if 'Prompt' in m and 'chars' in m:
            prompt_line = m
        if 'API calls' in m:
            final_msg = m
        if 'Tool call:' in m:
            api_count += 1
        if 'rror' in m:
            errors += 1

    print(f"\nTask {i+1}: {t['start'][:19]}")
    print(f"  Prompt: {prompt_line[:300]}")
    print(f"  Calls: {api_count}, Errors: {errors}")
    print(f"  End: {final_msg[:300]}")
