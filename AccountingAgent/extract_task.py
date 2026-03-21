import json, sys

with open(r'd:\NMiAI\AccountingAgent\rev33_latest.json', encoding='utf-8-sig') as f:
    data = json.load(f)

data.reverse()

start = sys.argv[1] if len(sys.argv) > 1 else '2026-03-21T13:01:53'
end = sys.argv[2] if len(sys.argv) > 2 else '2026-03-21T13:03:09'

for e in data:
    ts = e.get('timestamp', '')[:19]
    msg = e.get('textPayload', '')
    if ts >= start and ts <= end:
        print(f"{ts} | {msg[:600]}")
