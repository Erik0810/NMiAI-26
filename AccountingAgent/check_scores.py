"""
Check recent submission scores from the AINM competition page.

Connects to Chrome via CDP and scrapes the Recent Results section.
Can also click into individual results for detailed scoring breakdown.

Usage:
    python check_scores.py              # Show recent scores
    python check_scores.py --details N  # Click into Nth most recent result for details
    python check_scores.py --all        # Show all results with details
"""

import sys
import re
import time
import argparse
from playwright.sync_api import sync_playwright

CDP_URL = "http://localhost:9222"
SUBMIT_URL = "https://app.ainm.no/submit/tripletex"


def get_page(p):
    """Connect to Chrome and find the AINM page."""
    try:
        browser = p.chromium.connect_over_cdp(CDP_URL)
    except Exception as e:
        print(f"Cannot connect to Chrome at {CDP_URL}")
        print(f"Error: {e}")
        sys.exit(1)

    ctx = browser.contexts[0]
    page = None
    for pg in ctx.pages:
        if "ainm.no" in pg.url:
            page = pg
            break
    if not page:
        page = ctx.pages[0]
    return browser, page


def get_summary(page):
    """Get the summary stats (total score, rank, tasks attempted)."""
    body = page.locator("body").inner_text(timeout=10000)
    lines = body.split("\n")
    
    summary = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "Total Score" in line:
            # Next non-empty line is the score
            for j in range(i+1, min(i+4, len(lines))):
                val = lines[j].strip()
                if val and val[0].isdigit():
                    summary["total_score"] = val
                    break
        elif "Rank" in line and "#" not in line:
            for j in range(i+1, min(i+4, len(lines))):
                val = lines[j].strip()
                if val and "#" in val:
                    summary["rank"] = val
                    break
        elif line.startswith("#") and "of" in line:
            summary["rank"] = line
        elif "Tasks Attempted" in line:
            for j in range(i+1, min(i+4, len(lines))):
                val = lines[j].strip()
                if "/" in val:
                    summary["tasks_attempted"] = val
                    break
        elif "Submissions" in line and "daily" not in line.lower():
            for j in range(i+1, min(i+4, len(lines))):
                val = lines[j].strip()
                if val.isdigit():
                    summary["submissions"] = val
                    break
    
    return summary


def get_recent_results(page):
    """Parse the Recent Results section."""
    body = page.locator("body").inner_text(timeout=10000)
    lines = body.split("\n")
    
    results = []
    in_results = False
    current = {}
    
    for line in lines:
        line = line.strip()
        if "Recent Results" in line:
            in_results = True
            continue
        if not in_results:
            continue
        if not line:
            continue
        # Stop at footer
        if "Norwegian AI Championship" in line or "Astar Technologies" in line:
            break
        
        # Match "Task (X/Y)" pattern
        m = re.match(r"Task\s*\((\d+\.?\d*)/(\d+)\)", line)
        if m:
            if current:
                results.append(current)
            current = {
                "earned": float(m.group(1)),
                "max": int(m.group(2)),
            }
            continue
        
        # Match time and duration: "01:15 PM · 78.4s"
        m = re.match(r"(\d{1,2}:\d{2}\s*[AP]M)\s*[·╖]\s*([\d.]+)s", line)
        if m:
            current["time"] = m.group(1)
            current["duration"] = float(m.group(2))
            continue
        
        # Match percentage: "5/10 (50%)"
        m = re.match(r"(\d+\.?\d*)/(\d+)\s*\((\d+)%\)", line)
        if m:
            current["pct"] = int(m.group(3))
            continue
        
        # Match "Evaluating" or "Running"
        if "Evaluating" in line or "Running" in line or "Queued" in line:
            if current:
                current["status"] = line
            continue
    
    if current:
        results.append(current)
    
    return results


def click_result_details(page, index=0):
    """Click on a specific result to get detailed scoring breakdown."""
    # Refresh the page first
    page.goto(SUBMIT_URL, wait_until="networkidle")
    time.sleep(2)
    
    # Find all clickable result items
    # Look for elements that contain "Task (" pattern
    result_elements = page.locator("text=/Task \\(/").all()
    
    if index >= len(result_elements):
        print(f"Only {len(result_elements)} results visible, cannot access index {index}")
        return None
    
    # Click the result
    result_elements[index].click()
    time.sleep(2)
    
    # Get the detail text
    body = page.locator("body").inner_text(timeout=10000)
    return body


def print_scores(results, summary=None):
    """Pretty-print the scores."""
    if summary:
        print(f"{'='*60}")
        print(f"Total Score: {summary.get('total_score', '?')}  |  "
              f"Rank: {summary.get('rank', '?')}  |  "
              f"Tasks: {summary.get('tasks_attempted', '?')}  |  "
              f"Submissions: {summary.get('submissions', '?')}")
        print(f"{'='*60}")
    
    print(f"\n{'Recent Results':^60}")
    print(f"{'-'*60}")
    print(f"{'#':>3} {'Time':<10} {'Score':<12} {'Pct':>5} {'Duration':>10} {'Status':<10}")
    print(f"{'-'*60}")
    
    for i, r in enumerate(results):
        earned = r.get("earned", "?")
        mx = r.get("max", "?")
        pct = r.get("pct", "")
        tm = r.get("time", "?")
        dur = r.get("duration", "")
        status = r.get("status", "")
        
        score_str = f"{earned}/{mx}"
        pct_str = f"{pct}%" if pct else ""
        dur_str = f"{dur:.1f}s" if dur else ""
        
        # Color coding via markers
        marker = " "
        if pct == 100:
            marker = "OK"
        elif pct == 0:
            marker = "XX"
        elif pct and pct < 50:
            marker = "!!"
        
        print(f"{i+1:>3} {tm:<10} {score_str:<12} {pct_str:>5} {dur_str:>10} {marker} {status}")
    
    print(f"{'-'*60}")
    
    # Summary stats
    if results:
        perfect = sum(1 for r in results if r.get("pct") == 100)
        zeros = sum(1 for r in results if r.get("pct") == 0)
        avg_pct = sum(r.get("pct", 0) for r in results if r.get("pct") is not None) / len(results)
        print(f"\n  Perfect: {perfect}/{len(results)}  |  "
              f"Zeros: {zeros}/{len(results)}  |  "
              f"Avg: {avg_pct:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Check AINM submission scores")
    parser.add_argument("--details", type=int, default=-1,
                       help="Show details for Nth most recent result (0-based)")
    parser.add_argument("--refresh", action="store_true",
                       help="Refresh the page before reading")
    args = parser.parse_args()
    
    with sync_playwright() as p:
        browser, page = get_page(p)
        
        if args.refresh:
            page.goto(SUBMIT_URL, wait_until="networkidle")
            time.sleep(2)
        
        summary = get_summary(page)
        results = get_recent_results(page)
        print_scores(results, summary)
        
        if args.details >= 0:
            print(f"\n{'='*60}")
            print(f"Details for result #{args.details + 1}:")
            print(f"{'='*60}")
            detail_text = click_result_details(page, args.details)
            if detail_text:
                # Find the relevant section
                lines = detail_text.split("\n")
                in_detail = False
                for line in lines:
                    line = line.strip()
                    if "Score" in line or "Check" in line or "Points" in line or "correct" in line.lower():
                        in_detail = True
                    if in_detail and line:
                        print(line)
                    if in_detail and ("Norwegian AI Championship" in line):
                        break


if __name__ == "__main__":
    main()
