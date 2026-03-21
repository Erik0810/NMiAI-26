"""Extract full details for specific failing tasks."""
import json

with open('d:/NMiAI/AccountingAgent/all_logs.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

data.sort(key=lambda x: x['timestamp'])
print(f'Total entries: {len(data)}')

# Define time windows for the 3 zero-score tasks (UTC times)
# Task 17: Bank recon, completed 11:19:26, 210.9s -> started ~11:15:55
# Task 21: Unknown, completed 11:48:30, 103.3s -> started ~11:46:47
# Task 23: Ledger+project, completed 11:55:47, 237.1s -> started ~11:51:50

windows = [
    ('BANK_RECON_0/10', '2026-03-21T11:15:00', '2026-03-21T11:20:00'),
    ('UNKNOWN_0/10', '2026-03-21T11:46:00', '2026-03-21T11:49:00'),
    ('LEDGER_0/10', '2026-03-21T11:51:00', '2026-03-21T11:56:00'),
    # Also get 50% supplier invoices
    ('SUPPLIER_50%', '2026-03-21T11:39:00', '2026-03-21T11:41:30'),
    ('SUPPLIER_50%_2', '2026-03-21T12:15:00', '2026-03-21T12:17:00'),
]

for label, start, end in windows:
    print(f'\n{"="*80}')
    print(f' {label} (UTC {start} to {end})')
    print(f'{"="*80}')
    for d in data:
        ts = d['timestamp'][:22]
        if ts >= start and ts <= end:
            msg = d.get('textPayload', '')
            if msg and 'httpx' not in msg and 'google_genai' not in msg:
                # Show all non-http messages including prompts
                print(f'{ts} | {msg[:500]}')
