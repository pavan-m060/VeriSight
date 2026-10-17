import sys
import json
import re
from datetime import datetime
from playwright.sync_api import sync_playwright


# ============================================================
# HELPERS
# ============================================================

def extract_number(text):
    if not text:
        return 0

    match = re.search(r"([\d,]+)", text)

    if match:
        return int(
            match.group(1).replace(",", "")
        )

    return 0


# ============================================================
# RATING
# ============================================================

def extract_rating(review):

    try:
        rating_element = review.locator(
            "span.a-icon-alt"
        ).first

        text = rating_element.inner_text(
            timeout=3000
        ).strip()

        match = re.search(
            r"([\d.]+)",
            text
        )

        if match:
            return float(
                match.group(1)
            )

    except:
        pass

    return None


# ============================================================
# TITLE
# ============================================================

def extract_title(review):

    try:

        title_block = review.locator(
            '[data-hook="review-title"]'
        ).first

        text = title_block.inner_text(
            timeout=3000
        ).strip()

        # Remove unwanted badge text
        text = text.replace(
            "Verified Purchase",
            ""
        ).strip()

        return text

    except:
        pass

    # Fallback
    try:

        spans = review.locator(
            "span.a-text-bold"
        )

        for i in range(
            min(spans.count(), 10)
        ):

            text = spans.nth(i).inner_text(
                timeout=2000
            ).strip()

            if (
                text
                and text != "Verified Purchase"
            ):
                return text

    except:
        pass

    return ""


# ============================================================
# DATE
# ============================================================

def extract_date(review):

    try:

        date_element = review.locator(
            '[data-hook="review-date"]'
        ).first

        text = date_element.inner_text(
            timeout=3000
        ).strip()

        match = re.search(
            r"on (\d{1,2} \w+ \d{4})",
            text
        )

        if match:

            date = datetime.strptime(
                match.group(1),
                "%d %B %Y"
            )

            return date.strftime(
                "%Y-%m-%d"
            )

        return text

    except:
        return ""


# ============================================================
# VERIFIED PURCHASE
# ============================================================

def extract_verified(review):

    try:

        badge = review.locator(
            '[data-hook="avp-badge"]'
        ).first

        if badge.count() > 0:
            return True

    except:
        pass

    try:

        text = review.inner_text(
            timeout=3000
        )

        return (
            "Verified Purchase"
            in text
        )

    except:
        return False


# ============================================================
# REVIEW TEXT
# ============================================================

def extract_text(review):

    # Primary Amazon selector
    try:

        body = review.locator(
            '[data-hook="review-body"]'
        ).first

        if body.count() > 0:

            text = body.inner_text(
                timeout=3000
            ).strip()

            if text:
                return text

    except:
        pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    try:

        lines = review.inner_text(
            timeout=3000
        ).splitlines()

        lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        unwanted = {
            "Verified Purchase",
            "Helpful",
            "Report"
        }

        cleaned = []

        for line in lines:

            lower = line.lower()

            if line in unwanted:
                continue

            if (
                "people found this helpful"
                in lower
            ):
                continue

            if (
                "person found this helpful"
                in lower
            ):
                continue

            cleaned.append(line)

        # Find the review text after
        # the product/variant information.
        #
        # Usually the last meaningful paragraph
        # before helpful/report information.

        if cleaned:

            # Remove reviewer name
            if len(cleaned) > 0:
                cleaned = cleaned[1:]

            # Remove rating line
            cleaned = [
                x for x in cleaned
                if not re.match(
                    r"^\d+(\.\d+)?\s+out of 5 stars$",
                    x,
                    re.IGNORECASE
                )
            ]

            # Remove date
            cleaned = [
                x for x in cleaned
                if not x.startswith(
                    "Reviewed in "
                )
            ]

            # Remove variant information
            cleaned = [
                x for x in cleaned
                if not (
                    x.startswith("Size:")
                    or x.startswith("Colour:")
                    or x.startswith("Color:")
                )
            ]

            if cleaned:
                return " ".join(
                    cleaned
                )

    except:
        pass

    return ""


# ============================================================
# HELPFUL COUNT
# ============================================================

def extract_helpful(review):

    try:

        text = review.inner_text(
            timeout=3000
        )

        match = re.search(
            r"([\d,]+)\s+"
            r"(?:people|person)"
            r"\s+found this helpful",
            text,
            re.IGNORECASE
        )

        if match:

            return int(
                match.group(1)
                .replace(",", "")
            )

    except:
        pass

    return 0


# ============================================================
# SINGLE REVIEW
# ============================================================

def extract_single_review(review, index):

    data = {

        "review_number":
            index + 1,

        "rating":
            extract_rating(
                review
            ),

        "title":
            extract_title(
                review
            ),

        "text":
            extract_text(
                review
            ),

        "date":
            extract_date(
                review
            ),

        "verified_purchase":
            extract_verified(
                review
            ),

        "helpful_count":
            extract_helpful(
                review
            )
    }

    return data


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_reviews(url):

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page(
            viewport={
                "width": 1400,
                "height": 1000
            }
        )

        print()
        print("=" * 60)
        print("VERISIGHT AMAZON REVIEW EXTRACTOR")
        print("=" * 60)
        print()

        print("Opening Amazon page...")

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

        except Exception as e:

            print(
                f"Page loading warning: {e}"
            )

        # Give Amazon time to render
        page.wait_for_timeout(
            5000
        )

        # ====================================================
        # FIND REVIEWS
        # ====================================================

        review_blocks = page.locator(
            'div[data-hook="review"]'
        )

        review_count = (
            review_blocks.count()
        )

        print(
            f"Found {review_count} review blocks"
        )

        reviews = []

        # ====================================================
        # EXTRACT EACH REVIEW
        # ====================================================

        for i in range(
            review_count
        ):

            print(
                f"\nExtracting review "
                f"{i + 1}/{review_count}..."
            )

            try:

                review = (
                    review_blocks.nth(i)
                )

                data = (
                    extract_single_review(
                        review,
                        i
                    )
                )

                reviews.append(
                    data
                )

                print(
                    f"  Rating: "
                    f"{data['rating']}"
                )

                print(
                    f"  Title: "
                    f"{data['title']}"
                )

                print(
                    f"  Date: "
                    f"{data['date']}"
                )

                print(
                    f"  Verified: "
                    f"{data['verified_purchase']}"
                )

                print(
                    f"  Helpful: "
                    f"{data['helpful_count']}"
                )

                print(
                    f"  Text: "
                    f"{data['text'][:120]}"
                )

            except Exception as e:

                print(
                    f"  ERROR: {e}"
                )

        # ====================================================
        # PAGE SUMMARY
        # ====================================================

        result = {

            "url":
                url,

            "review_count":
                len(reviews),

            "reviews":
                reviews
        }

        # ====================================================
        # SAVE JSON
        # ====================================================

        output_file = (
            "results/"
            "amazon_reviews_structured.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)

        print(
            f"Reviews extracted: "
            f"{len(reviews)}"
        )

        print(
            f"Output file: "
            f"{output_file}"
        )

        print()

        # Keep browser open
        input(
            "Press ENTER to close browser..."
        )

        browser.close()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            'python scripts\\extract_amazon_reviews.py "AMAZON_URL"'
        )

        print()

        sys.exit(1)

    amazon_url = sys.argv[1]

    extract_reviews(
        amazon_url
    )