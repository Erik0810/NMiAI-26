"""Submit a task and check the score via CDP."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright
import time

def submit_and_check():
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
        time.sleep(2)
        
        # Fill in the endpoint URL and submit
        url_input = page.locator('input[placeholder*="https://"]').first
        url_input.fill('https://accounting-agent-t2rr5sny2q-ew.a.run.app/solve')
        time.sleep(0.5)
        
        submit_btn = page.locator('button:has-text("Submit")').first
        submit_btn.click()
        print('Submitted! Waiting for result...')
        
        # Wait for result to appear - poll every 10 seconds
        for i in range(18):  # 3 minutes max
            time.sleep(10)
            page.reload(wait_until='networkidle')
            time.sleep(2)
            items = page.locator('text=/Task \\(/').all()
            if items:
                txt = items[0].inner_text()
                # Check if the time is very recent (just submitted)
                print(f'  [{i*10}s] Latest: {txt}')

        # Final check
        page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
        time.sleep(3)
        items = page.locator('text=/Task \\(/').all()
        if items:
            print(f'\nLatest result: {items[0].inner_text()}')

if __name__ == '__main__':
    submit_and_check()
