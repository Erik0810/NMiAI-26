"""
Playwright script to submit tasks on the AINM Tripletex competition platform.

Approach: Connect to your existing Chrome via CDP to avoid Google's
"unsafe browser" block. You must start Chrome with remote debugging:

  1. Close all Chrome windows
  2. Run: chrome.exe --remote-debugging-port=9222
  3. Log into https://app.ainm.no in that Chrome window
  4. Then run: python submit_task.py

For subsequent runs, just: python submit_task.py --loop
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

SUBMIT_URL = "https://app.ainm.no/submit/tripletex"
ENDPOINT_URL = "https://accounting-agent-t2rr5sny2q-ew.a.run.app/solve"
CDP_URL = "http://localhost:9222"


def find_chrome():
    """Find Chrome executable."""
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "chrome.exe"  # hope it's on PATH


def submit_once(page):
    """Fill in the endpoint URL and click Submit."""
    page.goto(SUBMIT_URL, wait_until="networkidle")
    time.sleep(2)

    # Make sure the form is there
    endpoint_input = page.locator("#endpoint")
    endpoint_input.wait_for(state="visible", timeout=30000)

    # Clear and fill
    endpoint_input.click()
    endpoint_input.fill(ENDPOINT_URL)
    time.sleep(0.5)

    # Click submit
    page.locator('button[type="submit"]').click()
    print(f"[{time.strftime('%H:%M:%S')}] Submitted endpoint: {ENDPOINT_URL}")

    # Wait for response
    time.sleep(3)

    # Check for status messages
    try:
        body_text = page.locator("body").inner_text(timeout=5000)
        for line in body_text.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in ["submitted", "success", "queued", "running", "error", "failed", "daily submissions"]):
                print(f"  Status: {line[:120]}")
    except Exception:
        pass


def first_run():
    """First run: launch Chrome with debug port, user logs in, then submit."""
    chrome = find_chrome()
    profile_dir = Path(__file__).parent / "chrome_debug_profile"
    profile_dir.mkdir(exist_ok=True)

    print(f"Launching Chrome with remote debugging...")
    print(f"  Chrome: {chrome}")
    print(f"  Profile: {profile_dir}")
    print()

    # Launch Chrome with remote debugging
    proc = subprocess.Popen([
        chrome,
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_dir}",
        SUBMIT_URL,
    ])

    print("Chrome is open. Please:")
    print("  1. Log in with Google if needed")
    print("  2. Make sure you can see the submission form")
    print("  3. Press ENTER here when ready")
    print()
    input("Press ENTER when you're on the submission form... ")

    # Now connect via CDP
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        # Get the existing page
        context = browser.contexts[0]
        pages = context.pages
        page = None
        for pg in pages:
            if "ainm.no" in pg.url:
                page = pg
                break
        if not page:
            page = pages[0] if pages else context.new_page()

        submit_once(page)
        print("Done! Chrome stays open for future --loop runs.")
        print("Do NOT close Chrome if you want to use --loop mode.")
        # Don't close browser - keep Chrome running


def auto_submit():
    """Auto-submit by connecting to already-running Chrome."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            print(f"Cannot connect to Chrome at {CDP_URL}")
            print("Make sure Chrome is running with --remote-debugging-port=9222")
            print(f"Error: {e}")
            sys.exit(1)

        context = browser.contexts[0]
        pages = context.pages
        page = None
        for pg in pages:
            if "ainm.no" in pg.url:
                page = pg
                break
        if not page:
            page = pages[0] if pages else context.new_page()

        submit_once(page)
        # Don't close - keep Chrome running


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit tasks to AINM Tripletex")
    parser.add_argument("--loop", action="store_true", help="Connect to existing Chrome (skip login)")
    args = parser.parse_args()

    if args.loop:
        auto_submit()
    else:
        first_run()
