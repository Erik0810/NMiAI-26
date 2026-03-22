import json
import sys

with open(r'd:\NMiAI\AccountingAgent\logs_latest.json', encoding='utf-8-sig') as f:
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

if current_task:
    tasks.append(current_task)

# Mode selection
mode = sys.argv[1] if len(sys.argv) > 1 else 'summary'

if mode == 'full':
    # Full output for specific task number
    task_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if task_num < 1 or task_num > len(tasks):
        print(f"Task {task_num} not found (have {len(tasks)} tasks)")
        sys.exit(1)
    t = tasks[task_num - 1]
    print(f"TASK {task_num}: {t['start'][:19]}")
    print("=" * 80)
    for ts, m in t['msgs']:
        if len(m) > 1000:
            print(f"  [{ts}] {m[:1000]}... [TRUNCATED {len(m)} chars]")
        else:
            print(f"  [{ts}] {m}")
else:
    # Summary mode
    print(f"Found {len(tasks)} tasks\n")
    for i, t in enumerate(tasks):
        prompt = ''
        files = ''
        tool_calls = []
        errors_4xx = []
        final_msg = ''
        turns = 0
        api_calls = 0
        agent_errors = 0
        for ts, m in t['msgs']:
            if 'Prompt' in m and 'chars' in m:
                prompt = m.split('): ', 1)[-1] if '): ' in m else m
            if 'Files:' in m and '__main__' in m:
                files = m.split('Files: ')[-1] if 'Files: ' in m else ''
            if 'Tool call:' in m:
                call_name = m.split('Tool call: ')[-1][:120] if 'Tool call: ' in m else m[:120]
                tool_calls.append(call_name)
            if '4' in m and ('00' in m or '01' in m or '04' in m or '22' in m) and 'HTTP' in m and 'OK' not in m:
                errors_4xx.append(m)
            if 'HTTP/1.1 4' in m:
                errors_4xx.append(m.split('HTTP Request: ')[-1][:200] if 'HTTP Request: ' in m else m[:200])
            if 'Error' in m and 'tripletex' not in m and 'httpx' not in m:
                agent_errors += 1
            if 'Agent final message' in m:
                final_msg = m.split('Agent final message: ')[-1][:300] if 'Agent final message: ' in m else m[:300]
            if 'Turn ' in m or 'turn ' in m:
                turns += 1
            if 'API calls' in m:
                try:
                    api_calls = int(m.split('API calls')[0].split(',')[-1].strip().split()[-1])
                except:
                    pass
            if 'completed in' in m and 'turns' in m:
                try:
                    parts = m.split('completed in ')[1]
                    turns = int(parts.split(' turn')[0])
                    api_calls = int(parts.split(', ')[1].split(' API')[0])
                    agent_errors = int(parts.split(', ')[2].split(' error')[0])
                except:
                    pass

        print(f"\n{'='*80}")
        print(f"TASK {i+1}: {t['start'][:19]}")
        print(f"  Prompt: {prompt[:350]}")
        if files:
            print(f"  Files: {files}")
        print(f"  Turns: {turns}, API calls: {api_calls}, Errors: {agent_errors}")
        print(f"  Tool calls ({len(tool_calls)}):")
        for tc in tool_calls:
            print(f"    - {tc}")
        if errors_4xx:
            print(f"  HTTP 4xx errors ({len(errors_4xx)}):")
            for e in errors_4xx:
                print(f"    ! {e}")
        if final_msg:
            print(f"  Final: {final_msg}")
        else:
            print(f"  Final: [NO FINAL MESSAGE]")
    print()
