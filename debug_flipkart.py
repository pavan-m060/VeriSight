import re
from playwright.sync_api import sync_playwright

URL = "https://www.flipkart.com/red-tape-sneakers-men/p/itm7e392c192a605?pid=SHOHM9TGGHXYTTN4&marketplace=FLIPKART"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(
        viewport={"width": 1440, "height": 1000},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        )
    )

    print("Opening Flipkart...")
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(5000)

    print("Scrolling...")
    for i in range(10):
        page.mouse.wheel(0, 1800)
        page.wait_for_timeout(800)

    page.wait_for_timeout(3000)

    body = page.locator("body").inner_text(timeout=10000)

    print("\n========== PAGE TEXT CHECK ==========")

    keywords = [
        "Ratings and reviews",
        "Reviews",
        "review",
        "Verified Buyer",
        "Verified",
        "months ago",
        "days ago",
        "weeks ago",
        "Excellent",
        "Good quality",
        "Terrific",
        "Awesome",
        "Very good",
    ]

    lines = [x.strip() for x in body.splitlines() if x.strip()]

    for i, line in enumerate(lines):
        if any(k.lower() in line.lower() for k in keywords):
            start = max(0, i - 2)
            end = min(len(lines), i + 5)

            print("\n--- MATCH ---")
            for x in lines[start:end]:
                print(x)

    print("\n========== POSSIBLE REVIEW ELEMENTS ==========")

    selectors = [
        "div[data-testid]",
        "[class*='review']",
        "[class*='Review']",
        "[class*='rating']",
        "[class*='Rating']",
        "div",
    ]

    for selector in selectors:
        try:
            loc = page.locator(selector)
            count = loc.count()
            print(f"{selector}: {count}")

            shown = 0

            for i in range(min(count, 3000)):
                try:
                    txt = loc.nth(i).inner_text(timeout=100)
                    txt = re.sub(r"\s+", " ", txt).strip()

                    if not (30 <= len(txt) <= 1000):
                        continue

                    has_date = bool(re.search(
                        r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b",
                        txt,
                        re.I
                    ))

                    has_verified = bool(re.search(
                        r"verified",
                        txt,
                        re.I
                    ))

                    has_rating = bool(re.search(
                        r"\b[1-5](?:\.[0-9])?\b",
                        txt
                    ))

                    if has_date or has_verified:
                        print(f"\n[{selector} #{i}]")
                        print(txt[:900])
                        shown += 1

                        if shown >= 10:
                            break

                except Exception:
                    continue

        except Exception as e:
            print(selector, "ERROR:", e)

    page.screenshot(path="flipkart_debug.png", full_page=True)

    with open("flipkart_debug_text.txt", "w", encoding="utf-8") as f:
        f.write(body)

    print("\n========== FILES CREATED ==========")
    print("flipkart_debug.png")
    print("flipkart_debug_text.txt")

    print("\nBrowser will remain open for 10 seconds...")
    page.wait_for_timeout(10000)
    browser.close()
