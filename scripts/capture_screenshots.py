"""Capture full-resolution (1920x1080) screenshots of the app.

Start the app first, then run:

    python scripts/capture_screenshots.py [--base http://127.0.0.1:8730]

Uses the system Chrome via Playwright (channel="chrome"), so no Chromium
download is needed. Captures each page full-page at a 1920x1080 viewport.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [
    ("01-overview", "/"),
    ("02-chat-scratch", "/chat?user_id=maya&thread_id=demo&backend=scratch"),
    ("03-chat-langgraph", "/chat?user_id=maya&thread_id=demo&backend=langgraph"),
    ("04-entity-supersession", "/rung/3"),
    ("05-measure", "/evals"),
    ("06-data", "/data"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8730")
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=1,
        )
        for name, path in PAGES:
            page.goto(args.base + path, wait_until="load", timeout=30000)
            page.wait_for_timeout(400)  # let CSS settle
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
            print(f"captured {name}.png")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
