# api/amazon_scanner.py

import re
from playwright.sync_api import sync_playwright


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


# ============================================================
# RATING PARSER
# ============================================================

def parse_rating(text):

    if not text:
        return None

    text = clean_text(text)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*out\s*of\s*5",
        r"(\d+(?:\.\d+)?)\s*stars?",
        r"(\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            try:

                rating = float(
                    match.group(1)
                )

                if 0 <= rating <= 5:
                    return rating

            except Exception:
                pass

    return None


# ============================================================
# DATE PARSER
# ============================================================

def parse_date(text):

    if not text:
        return ""

    text = clean_text(text)

    match = re.search(
        r"on\s+(.+)$",
        text,
        re.IGNORECASE
    )

    if match:
        return clean_text(
            match.group(1)
        )

    return text


# ============================================================
# HELPFUL COUNT PARSER
# ============================================================

def parse_helpful_count(text):

    if not text:
        return 0

    text = clean_text(text)

    match = re.search(
        r"(\d+)\s+people?\s+found\s+this\s+helpful",
        text,
        re.IGNORECASE
    )

    if match:

        try:
            return int(
                match.group(1)
            )
        except Exception:
            pass

    if re.search(
        r"one\s+person\s+found\s+this\s+helpful",
        text,
        re.IGNORECASE
    ):
        return 1

    return 0


# ============================================================
# PRODUCT INFORMATION
# ============================================================

def extract_amazon_product_info(page):

    overall_rating = None
    total_ratings = None

    # --------------------------------------------------------
    # Overall rating
    # --------------------------------------------------------

    rating_selectors = [
        "#acrPopover",
        "[data-hook='average-star-rating']",
        "[data-hook='rating-out-of-text']",
        ".a-icon-alt"
    ]

    for selector in rating_selectors:

        try:

            elements = page.locator(
                selector
            )

            count = elements.count()

            for i in range(
                min(count, 10)
            ):

                try:

                    text = elements.nth(i).inner_text(
                        timeout=2000
                    )

                except Exception:

                    try:

                        text = elements.nth(i).get_attribute(
                            "title"
                        )

                    except Exception:

                        text = ""

                rating = parse_rating(
                    text
                )

                if rating is not None:

                    overall_rating = rating
                    break

            if overall_rating is not None:
                break

        except Exception:
            continue

    # --------------------------------------------------------
    # Total ratings
    # --------------------------------------------------------

    rating_count_selectors = [
        "#acrCustomerReviewText",
        "[data-hook='total-review-count']"
    ]

    for selector in rating_count_selectors:

        try:

            elements = page.locator(
                selector
            )

            count = elements.count()

            for i in range(
                min(count, 10)
            ):

                try:

                    text = elements.nth(i).inner_text(
                        timeout=2000
                    )

                except Exception:

                    text = ""

                match = re.search(
                    r"([\d,]+)",
                    text
                )

                if match:

                    total_ratings = int(
                        match.group(1).replace(
                            ",",
                            ""
                        )
                    )

                    break

            if total_ratings is not None:
                break

        except Exception:
            continue

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if (
        overall_rating is None
        or total_ratings is None
    ):

        try:

            body_text = page.locator(
                "body"
            ).inner_text()

            body_text = clean_text(
                body_text
            )

            if overall_rating is None:

                patterns = [
                    r"(\d+(?:\.\d+)?)\s+out\s+of\s+5\s+stars?",
                    r"(\d+(?:\.\d+)?)\s+out\s+of\s+5"
                ]

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        body_text,
                        re.IGNORECASE
                    )

                    if match:

                        overall_rating = float(
                            match.group(1)
                        )

                        break

            if total_ratings is None:

                patterns = [
                    r"([\d,]+)\s+ratings?",
                    r"([\d,]+)\s+global\s+ratings?"
                ]

                for pattern in patterns:

                    match = re.search(
                        pattern,
                        body_text,
                        re.IGNORECASE
                    )

                    if match:

                        total_ratings = int(
                            match.group(1).replace(
                                ",",
                                ""
                            )
                        )

                        break

        except Exception:
            pass

    print(
        f"Overall Rating: {overall_rating}"
    )

    print(
        f"Total Ratings: {total_ratings}"
    )

    return {
        "overall_rating": overall_rating,
        "total_ratings": total_ratings
    }


# ============================================================
# AMAZON REVIEWS
# ============================================================

def extract_amazon_reviews(url):

    print("\nExtracting Amazon reviews...")

    reviews = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 1000
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(
                5000
            )

            # ------------------------------------------------
            # Find review blocks
            # ------------------------------------------------

            review_blocks = page.locator(
                'div[data-hook="review"]'
            )

            count = review_blocks.count()

            print(
                f"Review blocks found: {count}"
            )

            # ------------------------------------------------
            # Process reviews
            # ------------------------------------------------

            for i in range(count):

                if len(reviews) >= 8:
                    break

                block = review_blocks.nth(i)

                try:

                    block_text = block.inner_text(
                        timeout=5000
                    )

                except Exception as e:

                    print(
                        f"Review {i + 1}: "
                        f"could not read block"
                    )

                    continue

                lines = [
                    clean_text(line)
                    for line in block_text.splitlines()
                    if clean_text(line)
                ]

                if not lines:
                    continue

                # ------------------------------------------------
                # Rating
                # ------------------------------------------------

                rating = None

                rating_match = re.search(
                    r"(\d+(?:\.\d+)?)\s+out\s+of\s+5\s+stars?",
                    block_text,
                    re.IGNORECASE
                )

                if rating_match:

                    rating = float(
                        rating_match.group(1)
                    )

                # ------------------------------------------------
                # Date
                # ------------------------------------------------

                date = ""

                date_match = re.search(
                    r"Reviewed\s+in\s+.*?\s+on\s+(.+?)(?:\n|$)",
                    block_text,
                    re.IGNORECASE
                )

                if date_match:

                    date = clean_text(
                        date_match.group(1)
                    )

                # ------------------------------------------------
                # Verified purchase
                # ------------------------------------------------

                verified_purchase = bool(
                    re.search(
                        r"verified\s+purchase",
                        block_text,
                        re.IGNORECASE
                    )
                )

                # ------------------------------------------------
                # Helpful count
                # ------------------------------------------------

                helpful_count = parse_helpful_count(
                    block_text
                )

                # ------------------------------------------------
                # REVIEW TEXT
                #
                # We use the structure of the actual Amazon
                # review block:
                #
                # Reviewed...
                # Size...
                # Verified Purchase
                #
                # Review text
                #
                # 2 people found this helpful
                # Helpful
                # Report
                # ------------------------------------------------

                review_text = ""

                verified_index = -1

                for j, line in enumerate(lines):

                    if re.search(
                        r"verified\s+purchase",
                        line,
                        re.IGNORECASE
                    ):

                        verified_index = j
                        break

                helpful_index = len(lines)

                for j, line in enumerate(lines):

                    if re.search(
                        r"\d+\s+people?\s+found\s+this\s+helpful",
                        line,
                        re.IGNORECASE
                    ):

                        helpful_index = j
                        break

                    if re.search(
                        r"one\s+person\s+found\s+this\s+helpful",
                        line,
                        re.IGNORECASE
                    ):

                        helpful_index = j
                        break

                    if line.lower() == "helpful":
                        helpful_index = j
                        break

                    if line.lower() == "report":
                        helpful_index = j
                        break

                # ------------------------------------------------
                # Primary extraction
                # ------------------------------------------------

                if verified_index >= 0:

                    start = verified_index + 1

                    candidate_lines = lines[
                        start:helpful_index
                    ]

                    # Remove metadata lines such as:
                    # Size: ...
                    # Colour: ...
                    # Color: ...
                    filtered = []

                    for line in candidate_lines:

                        if re.match(
                            r"^(size|colour|color|style)\s*:",
                            line,
                            re.IGNORECASE
                        ):
                            continue

                        if re.search(
                            r"verified\s+purchase",
                            line,
                            re.IGNORECASE
                        ):
                            continue

                        filtered.append(
                            line
                        )

                    if filtered:

                        review_text = " ".join(
                            filtered
                        )

                # ------------------------------------------------
                # Fallback: review body selector
                # ------------------------------------------------

                if not review_text:

                    try:

                        body_locator = block.locator(
                            '[data-hook="review-body"]'
                        )

                        if body_locator.count() > 0:

                            review_text = clean_text(
                                body_locator.first.inner_text(
                                    timeout=3000
                                )
                            )

                    except Exception:
                        pass

                # ------------------------------------------------
                # Final fallback
                # ------------------------------------------------

                if not review_text:

                    try:

                        # Look for text after date line
                        # and before helpful controls.

                        date_line_index = -1

                        for j, line in enumerate(lines):

                            if re.search(
                                r"reviewed\s+in\s+",
                                line,
                                re.IGNORECASE
                            ):

                                date_line_index = j
                                break

                        if date_line_index >= 0:

                            candidate_lines = lines[
                                date_line_index + 1:
                                helpful_index
                            ]

                            filtered = []

                            for line in candidate_lines:

                                if re.match(
                                    r"^(size|colour|color|style)\s*:",
                                    line,
                                    re.IGNORECASE
                                ):
                                    continue

                                if re.search(
                                    r"verified\s+purchase",
                                    line,
                                    re.IGNORECASE
                                ):
                                    continue

                                filtered.append(
                                    line
                                )

                            if filtered:

                                review_text = " ".join(
                                    filtered
                                )

                    except Exception:
                        pass

                # ------------------------------------------------
                # Skip if still empty
                # ------------------------------------------------

                if not review_text:

                    print(
                        f"Review {i + 1}: "
                        "review text not detected"
                    )

                    continue

                # ------------------------------------------------
                # Create review
                # ------------------------------------------------

                review = {
                    "review_number": len(reviews) + 1,
                    "rating": rating,
                    "text": review_text,
                    "date": date,
                    "verified_purchase": verified_purchase,
                    "helpful_count": helpful_count
                }

                reviews.append(
                    review
                )

                print(
                    f"Review {review['review_number']}: "
                    f"{review}"
                )

            print(
                f"Total reviews extracted: "
                f"{len(reviews)}"
            )

            return reviews

        finally:

            browser.close()


# ============================================================
# CUSTOMERS SAY
# ============================================================

def extract_customers_say_topics(url):

    print("\nExtracting Customers Say topics...")

    topics = [
        "Quality",
        "Comfort",
        "Value for money",
        "Appearance"
    ]

    results = {}

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 1000
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(
                5000
            )

            # ------------------------------------------------
            # Scroll to Customers Say
            # ------------------------------------------------

            for _ in range(6):

                try:

                    page.mouse.wheel(
                        0,
                        1000
                    )

                    page.wait_for_timeout(
                        500
                    )

                except Exception:
                    break

            # ------------------------------------------------
            # Read page text
            # ------------------------------------------------

            try:

                body_text = page.locator(
                    "body"
                ).inner_text()

                body_text = clean_text(
                    body_text
                )

            except Exception:

                body_text = ""

            # ====================================================
            # Process topics
            # ====================================================

            for topic in topics:

                result = {
                    "found": False,
                    "clicked": False,
                    "summary": "",
                    "positive": None,
                    "negative": None
                }

                # ------------------------------------------------
                # Match:
                #
                # Quality(20)
                # Comfort(9)
                # Value for money(8)
                # Appearance(6)
                # ------------------------------------------------

                topic_pattern = re.compile(
                    rf"{re.escape(topic)}"
                    rf"\s*\(\s*(\d+)\s*\)",
                    re.IGNORECASE
                )

                topic_match = topic_pattern.search(
                    body_text
                )

                if not topic_match:

                    print(
                        f"{topic}: not found"
                    )

                    results[topic] = result

                    continue

                result["found"] = True

                mentions = int(
                    topic_match.group(1)
                )

                # ------------------------------------------------
                # Try to click
                # ------------------------------------------------

                try:

                    candidates = page.locator(
                        "text=" + topic
                    )

                    candidate_count = candidates.count()

                    for i in range(
                        min(candidate_count, 20)
                    ):

                        try:

                            element = candidates.nth(i)

                            if element.is_visible():

                                element.scroll_into_view_if_needed()

                                try:

                                    element.click(
                                        timeout=3000
                                    )

                                except Exception:

                                    element.click(
                                        timeout=3000,
                                        force=True
                                    )

                                result["clicked"] = True

                                page.wait_for_timeout(
                                    1500
                                )

                                break

                        except Exception:
                            continue

                except Exception:
                    pass

                # ------------------------------------------------
                # Read updated page
                # ------------------------------------------------

                try:

                    updated_text = page.locator(
                        "body"
                    ).inner_text()

                    updated_text = clean_text(
                        updated_text
                    )

                except Exception:

                    updated_text = body_text

                # ------------------------------------------------
                # Only accept positive/negative values when
                # they are explicitly associated with this topic.
                # ------------------------------------------------

                detailed_pattern = re.search(
                    rf"(\d+)\s+customers?\s+mention\s+"
                    rf"{re.escape(topic.lower())}"
                    rf".{{0,150}}?"
                    rf"(\d+)\s+positive"
                    rf".{{0,100}}?"
                    rf"(\d+)\s+negative",
                    updated_text,
                    re.IGNORECASE
                )

                if detailed_pattern:

                    detailed_mentions = int(
                        detailed_pattern.group(1)
                    )

                    positive = int(
                        detailed_pattern.group(2)
                    )

                    negative = int(
                        detailed_pattern.group(3)
                    )

                    result["positive"] = positive
                    result["negative"] = negative

                    result["summary"] = (
                        f"{detailed_mentions} customers "
                        f"mention {topic.lower()}, "
                        f"{positive} positive, "
                        f"{negative} negative"
                    )

                else:

                    # Amazon only exposed the topic count.
                    result["summary"] = (
                        f"{mentions} customers "
                        f"mention {topic.lower()}"
                    )

                print(
                    f"{topic}: "
                    f"{result['summary']}"
                )

                results[topic] = result

            return results

        finally:

            browser.close()


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_url = (
        "https://www.amazon.in/"
        "layasa-Womens-Sneaker-Stylish-Women/"
        "dp/B0FDR6X8B5/"
    )

    print("=" * 70)
    print("VERISIGHT AMAZON SCANNER")
    print("=" * 70)

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        page.goto(
            test_url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        page.wait_for_timeout(
            5000
        )

        print("\nPRODUCT INFORMATION")
        print("-" * 40)

        print(
            extract_amazon_product_info(page)
        )

        browser.close()

    print("\nREVIEWS")
    print("-" * 40)

    reviews = extract_amazon_reviews(
        test_url
    )

    print(
        f"\nExtracted {len(reviews)} reviews"
    )

    print("\nCUSTOMERS SAY")
    print("-" * 40)

    customers_say = extract_customers_say_topics(
        test_url
    )

    print(
        customers_say
    )