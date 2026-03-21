"""Submit N tasks and wait for results."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright
import time

ENDPOINT = 'https://accounting-agent-t2rr5sny2q-ew.a.run.app/solve'
N = 2

def main():
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

        # Submit N tasks
        for i in range(N):
            page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
            time.sleep(1)
            url_input = page.locator('input[placeholder*="https://"]').first
            url_input.fill(ENDPOINT)
            time.sleep(0.3)
            submit_btn = page.locator('button:has-text("Submit")').first
            submit_btn.click()
            print(f'Submitted {i+1}/{N}', flush=True)
            time.sleep(3)
        
        print(f'All {N} submitted! Waiting 3 minutes for results...', flush=True)
        time.sleep(180)
        
        # Check scores
        page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
        time.sleep(3)
        
        body = page.locator('body').inner_text()
        for line in body.split('\n'):
            line = line.strip()
            if any(x in line for x in ['Total Score', 'Rank', 'Tasks Attempted', 'Submissions']):
                print(line, flush=True)
        
        items = page.locator('text=/Task \\(/').all()
        for i, item in enumerate(items[:8]):
            txt = item.inner_text()
            print(f'{i+1}. {txt}', flush=True)

if __name__ == '__main__':
    main()
