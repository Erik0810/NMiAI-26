"""Submit N tasks one at a time with small gaps."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ENDPOINT = 'https://accounting-agent-t2rr5sny2q-ew.a.run.app/solve'

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp('http://localhost:9222')
    ctx = browser.contexts[0]
    page = None
    for pg in ctx.pages:
        if 'ainm.no' in pg.url:
            page = pg
            break
    if not page:
        page = ctx.pages[0]

    for i in range(N):
        try:
            page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
            time.sleep(1)
            url_input = page.locator('input[placeholder*="https://"]').first
            url_input.fill(ENDPOINT)
            time.sleep(0.3)
            submit_btn = page.locator('button:has-text("Submit")').first
            submit_btn.click()
            print(f'Submitted {i+1}/{N}', flush=True)
            if i < N - 1:
                print(f'  Waiting 60s before next submission...', flush=True)
                time.sleep(60)
        except Exception as e:
            print(f'Error on {i+1}/{N}: {e}', flush=True)
            time.sleep(5)

    print(f'Done! Submitted {N} tasks.', flush=True)
