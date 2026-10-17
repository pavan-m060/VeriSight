import sys
from playwright.sync_api import sync_playwright


def main(url):

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=False)

        page = browser.new_page(
            viewport={"width": 1400, "height": 1000}
        )

        print("Opening Amazon...")

        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        page.wait_for_timeout(5000)

        reviews = page.locator(
            'div[data-hook="review"]'
        )

        count = reviews.count()

        print(f"\nFound {count} review blocks\n")

        for i in range(count):

            review = reviews.nth(i)

            print("=" * 70)
            print(f"REVIEW {i + 1}")
            print("=" * 70)

            # Get ALL visible text inside this review block
            try:

                text = review.inner_text(
                    timeout=5000
                )

                print(text[:3000])

            except Exception as e:

                print("ERROR:", e)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)

        input("\nPress ENTER to close...")

        browser.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            'Usage: python scripts\\test_amazon_reviews.py "URL"'
        )

        sys.exit(1)

    main(sys.argv[1])