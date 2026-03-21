"""Rapidly submit tasks and check scores."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright
import time
import argparse

ENDPOINT = 'https://accounting-agent-t2rr5sny2q-ew.a.run.app/solve'

def get_page(browser):
    ctx = browser.contexts[0]
    for pg in ctx.pages:
        if 'ainm.no' in pg.url:
            return pg
    return ctx.pages[0]

def submit_one(page):
    """Submit one task and return immediately."""
    page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
    time.sleep(1)
    url_input = page.locator('input[placeholder*="https://"]').first
    url_input.fill(ENDPOINT)
    time.sleep(0.3)
    submit_btn = page.locator('button:has-text("Submit")').first
    submit_btn.click()
    return time.time()

def get_latest_scores(page, count=5):
    """Get the N most recent scores."""
    page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
    time.sleep(2)
    items = page.locator('text=/Task \\(/').all()
    results = []
    for item in items[:count]:
        txt = item.inner_text()
        parent = item.locator('..')
        detail_el = parent.locator('..').locator('span, p')
        details = []
        results.append(txt)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3, help='Number of submissions')
    parser.add_argument('--delay', type=int, default=90, help='Seconds between submissions')
    args = parser.parse_args()
    
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp('http://localhost:9222')
        page = get_page(browser)
        
        for i in range(args.n):
            t0 = submit_one(page)
            print(f'[{i+1}/{args.n}] Submitted at {time.strftime("%H:%M:%S")}')
            
            # Wait for result
            time.sleep(args.delay)
            
            # Check scores
            page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
            time.sleep(2)
            items = page.locator('text=/Task \\(/').all()
            if items:
                print(f'  Latest result: {items[0].inner_text()}')
            
            if i < args.n - 1:
                time.sleep(5)  # Brief pause between submissions
        
        # Final score check
        print('\n--- Final scores (latest 10) ---')
        page.goto('https://app.ainm.no/submit/tripletex', wait_until='networkidle')
        time.sleep(3)
        
        # Get summary
        body = page.locator('body').inner_text()
        for line in body.split('\n'):
            line = line.strip()
            if any(x in line for x in ['Total Score', 'Rank', 'Tasks Attempted', 'Submissions']):
                print(f'  {line}')
        
        items = page.locator('text=/Task \\(/').all()
        for i, item in enumerate(items[:10]):
            txt = item.inner_text()
            parent_text = item.locator('..').inner_text()
            # Extract time and percentage
            print(f'  {i+1}. {parent_text.strip()[:80]}')

if __name__ == '__main__':
    main()
