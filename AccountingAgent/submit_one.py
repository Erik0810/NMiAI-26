"""Submit 1 test task."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright
import time

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

    page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
    time.sleep(1)
    url_input = page.locator('input[placeholder*="https://"]').first
    url_input.fill(ENDPOINT)
    time.sleep(0.3)
    submit_btn = page.locator('button:has-text("Submit")').first
    submit_btn.click()
    print('Submitted 1 test task!', flush=True)
