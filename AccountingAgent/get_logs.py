"""Download rev37 logs and do thorough analysis."""
import subprocess, json, re

GCLOUD = r"C:\Users\erikk\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

filter_str = 'resource.type="cloud_run_revision" AND resource.labels.revision_name="accounting-agent-00037-hac"'

cmd = [GCLOUD, "logging", "read", filter_str, 
       "--project", "ainm26osl-764", "--limit", "3000", 
       "--format", "json", "--freshness", "24h"]

print("Fetching logs (may take a minute)...", flush=True)
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
entries = json.loads(result.stdout)
print(f"Got {len(entries)} entries")

with open("rev37_logs.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)

# Reverse to chronological order
entries.reverse()

# Group into tasks by POST /solve boundaries
tasks = []
current = []
for e in entries:
    t = e.get("textPayload", "")
    if not t:
        continue
    if 'POST /solve' in t and '200 OK' in t:
        # This is end-of-request marker
        current.append(t)
        tasks.append(current)
        current = []
    else:
        current.append(t)
if current:
    tasks.append(current)

print(f"\nFound {len(tasks)} request groups")

for i, task_lines in enumerate(tasks):
    prompt = ""
    completion = ""
    tools = []
    errors = []
    
    for line in task_lines:
        if "Prompt (" in line:
            m = re.search(r"chars\):\s*(.+)", line)
            if m:
                prompt = m.group(1)[:250]
        if "Task completed" in line:
            completion = line
        if "Tool call:" in line:
            m = re.search(r"Tool call: (\w+)", line)
            if m:
                tools.append(m.group(1))
        if "422" in line and "WARNING" in line:
            m = re.search(r'"message":"([^"]+)"', line)
            if m:
                errors.append(m.group(1)[:200])
        if "validation" in line.lower() and "error" in line.lower():
            m = re.search(r'"validationMessages":\[([^\]]+)\]', line)
            if m:
                errors.append(m.group(1)[:200])
    
    if not prompt and not completion:
        continue
    
    dur = calls = errs = "?"
    if completion:
        m = re.search(r"in ([\d.]+)s.*?API calls: (\d+).*?Errors: (\d+)", completion)
        if m:
            dur, calls, errs = m.group(1), m.group(2), m.group(3)
    
    print(f"\n[Task {i+1}] {dur}s | {calls} calls | {errs} errors")
    if prompt:
        print(f"  PROMPT: {prompt}")
    print(f"  TOOLS({len(tools)}): {' -> '.join(tools[:15])}")
    if errors:
        for e in set(errors):
            print(f"  ERR: {e}")

# All unique errors
print(f"\n{'='*80}")
print("ALL UNIQUE ERRORS")
print(f"{'='*80}")
all_err = set()
for e in entries:
    t = e.get("textPayload", "")
    if ("422" in t or "400" in t) and "WARNING" in t:
        m = re.search(r'"message":"([^"]+)"', t)
        if m:
            all_err.add(m.group(1)[:200])
for err in sorted(all_err):
    print(f"  - {err}")
