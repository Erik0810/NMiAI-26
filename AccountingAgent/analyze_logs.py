"""Get rev37 logs via gcloud CLI and analyze them."""
import subprocess, json, re, sys

GCLOUD = r"C:\Users\erikk\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

filter_str = (
    'resource.type="cloud_run_revision" '
    'AND resource.labels.revision_name="accounting-agent-00037-hac"'
)

cmd = [
    GCLOUD, "logging", "read", filter_str,
    "--project", "ainm26osl-764",
    "--limit", "1000",
    "--format", "json",
    "--freshness", "6h"
]

print("Fetching logs...", flush=True)
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
if result.returncode != 0:
    print(f"STDERR: {result.stderr[:500]}")
    sys.exit(1)

entries = json.loads(result.stdout)
print(f"Got {len(entries)} entries")

# Extract all text payloads, sorted by time (oldest first)
texts = []
for e in reversed(entries):
    t = e.get("textPayload", "")
    if t:
        texts.append(t)

# Group into task sections based on "Prompt" or "Starting task" markers
task_sections = []
current = []
for line in texts:
    if ("Prompt (" in line and "chars)" in line) or "=== Starting task" in line:
        if current:
            task_sections.append(current)
        current = [line]
    else:
        current.append(line)
if current:
    task_sections.append(current)

print(f"\n{'='*100}")
print(f"FOUND {len(task_sections)} TASK EXECUTION(S)")
print(f"{'='*100}")

for i, section in enumerate(task_sections):
    prompt = ""
    completion = ""
    errors_422 = []
    tool_calls = []
    
    for line in section:
        if "Prompt (" in line:
            m = re.search(r'chars\):\s*(.+)', line)
            if m:
                prompt = m.group(1)[:200]
        if "Task completed" in line:
            completion = line
        if "Tool call:" in line:
            m = re.search(r'Tool call: (\w+)\(', line)
            if m:
                tool_calls.append(m.group(1))
        if "422" in line and ("WARNING" in line or "status_code" in line):
            m = re.search(r'"message":"([^"]+)"', line)
            msg = m.group(1) if m else "unknown 422"
            errors_422.append(msg[:150])
    
    dur = calls = errs = "?"
    if completion:
        m = re.search(r'in ([\d.]+)s.*?API calls: (\d+).*?Errors: (\d+)', completion)
        if m:
            dur, calls, errs = m.group(1), m.group(2), m.group(3)
    
    status = "OK" if errs == "0" else f"!{errs} ERRS"
    print(f"\n[{i+1}] {dur}s | {calls} calls | {status}")
    if prompt:
        print(f"    PROMPT: {prompt}")
    if tool_calls:
        print(f"    TOOLS: {' -> '.join(tool_calls[:12])}")
    if errors_422:
        for e in set(errors_422):
            print(f"    422: {e}")

# Summary
print(f"\n{'='*100}")
print("ALL UNIQUE ERROR MESSAGES")
print(f"{'='*100}")
all_errors = set()
for line in texts:
    if ("422" in line or "400" in line) and ("WARNING" in line or "status_code" in line):
        m = re.search(r'"message":"([^"]+)"', line)
        if m:
            all_errors.add(m.group(1)[:200])
for e in sorted(all_errors):
    print(f"  - {e}")

print(f"\n{'='*100}")
print("ALL PROMPTS SEEN")
print(f"{'='*100}")
for line in texts:
    if "Prompt (" in line:
        m = re.search(r'chars\):\s*(.+)', line)
        if m:
            print(f"  {m.group(1)[:200]}")
