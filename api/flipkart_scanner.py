# ============================================================
# VERISIGHT FLIPKART SCANNER
# ============================================================

import re
from typing import List, Dict, Optional

from playwright.sync_api import sync_playwright


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_rating(text: str) -> Optional[float]:

    if not text:
        return None

    match = re.search(
        r"(?<!\d)([1-5](?:\.\d)?)\s*(?:★|⭐)?",
        text
    )

    if match:

        value = float(match.group(1))

        if 1 <= value <= 5:
            return value

    return None


def parse_helpful_numbers(text: str):

    """
    Flipkart review cards commonly expose:

        helpful_count
        not_helpful_count

    Example:

        8
        3

    """

    numbers = re.findall(
        r"(?<!\d)\d+(?!\d)",
        text
    )

    if len(numbers) >= 2:

        try:

            return (
                int(numbers[-2]),
                int(numbers[-1])
            )

        except ValueError:
            pass

    if len(numbers) == 1:

        try:
            return int(numbers[0]), 0
        except ValueError:
            pass

    return 0, 0


# ============================================================
# PRODUCT INFORMATION
# ============================================================

def extract_flipkart_product_info(page):

    rating = None
    total_ratings = None

    body_text = ""

    try:
        body_text = page.locator("body").inner_text()
    except Exception:
        pass

    # --------------------------------------------------------
    # Overall rating
    # --------------------------------------------------------

    selectors = [
        "div._3LWZlK",
        "div[class*='_3LWZlK']",
        "div[class*='rating']"
    ]

    for selector in selectors:

        try:

            elements = page.locator(selector)

            count = elements.count()

            for i in range(min(count, 20)):

                text = clean_text(
                    elements.nth(i).inner_text()
                )

                value = parse_rating(text)

                if value is not None:

                    rating = value
                    break

            if rating is not None:
                break

        except Exception:
            continue

    # --------------------------------------------------------
    # Fallback from body text
    # --------------------------------------------------------

    if rating is None:

        match = re.search(
            r"Ratings and reviews\s+"
            r"([1-5](?:\.\d)?)",
            body_text,
            re.I
        )

        if match:

            rating = float(
                match.group(1)
            )

    # --------------------------------------------------------
    # Total ratings
    # --------------------------------------------------------

    patterns = [

        r"based on\s+([\d,]+)\s+ratings",

        r"([\d,]+)\s+ratings\s+by",

        r"([\d,]+)\s+ratings"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            body_text,
            re.I
        )

        if match:

            try:

                total_ratings = int(
                    match.group(1).replace(",", "")
                )

                break

            except ValueError:
                pass

    return {

        "overall_rating": rating,

        "total_ratings":
            total_ratings

    }


# ============================================================
# FIND REVIEW CONTAINERS
# ============================================================

def find_flipkart_review_blocks(page):

    candidates = [

        "div[class*='col'] div[class*='row']",

        "div[class*='review']",

        "div[class*='Review']",

        "div._27M-vq",

        "div._16PBlm"

    ]

    best_blocks = []

    best_count = 0

    for selector in candidates:

        try:

            locator = page.locator(selector)

            count = locator.count()

            if count > best_count:

                # Avoid selecting thousands of generic divs
                if count <= 100:

                    best_blocks = locator
                    best_count = count

        except Exception:
            continue

    return best_blocks


# ============================================================
# EXTRACT REVIEWS
# ============================================================

def extract_flipkart_reviews(
    url: str
) -> List[Dict]:

    print()
    print("Extracting Flipkart reviews...")
    print()

    reviews = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 1200
            }
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(
                7000
            )

            # ------------------------------------------------
            # Scroll to reviews
            # ------------------------------------------------

            try:

                text_locator = page.get_by_text(
                    "Ratings and reviews",
                    exact=False
                )

                if text_locator.count() > 0:

                    text_locator.first.scroll_into_view_if_needed()

                    page.wait_for_timeout(
                        2000
                    )

            except Exception:
                pass

            # Additional scroll
            try:

                page.mouse.wheel(
                    0,
                    4500
                )

                page.wait_for_timeout(
                    2000
                )

            except Exception:
                pass

            body_text =
                page.locator("body").inner_text()

            # ------------------------------------------------
            # Look for review titles
            # ------------------------------------------------

            review_titles = [
                "Perfect product!",
                "Delightful",
                "Awesome",
                "Terrific purchase",
                "Mind-blowing purchase",
                "Simply awesome"
            ]

            # ------------------------------------------------
            # Try known Flipkart review structures
            # ------------------------------------------------

            candidate_selectors = [

                "div._27M-vq",

                "div._16PBlm",

                "div[class*='review']",

                "div[class*='Review']"

            ]

            blocks = []

            for selector in candidate_selectors:

                try:

                    locator = page.locator(
                        selector
                    )

                    count = locator.count()

                    if count >= 1 and count <= 100:

                        for i in range(count):

                            block =
                                locator.nth(i)

                            text =
                                clean_text(
                                    block.inner_text()
                                )

                            if (
                                "Verified Buyer"
                                in text
                                and len(text) > 30
                            ):

                                blocks.append(
                                    block
                                )

                except Exception:
                    continue

            # Remove duplicates
            unique_texts = set()
            unique_blocks = []

            for block in blocks:

                try:

                    text =
                        clean_text(
                            block.inner_text()
                        )

                    if text in unique_texts:
                        continue

                    unique_texts.add(text)

                    unique_blocks.append(
                        block
                    )

                except Exception:
                    continue

            print(
                "Review candidates found:",
                len(unique_blocks)
            )

            # ------------------------------------------------
            # Extract each review
            # ------------------------------------------------

            for index, block in enumerate(
                unique_blocks,
                start=1
            ):

                try:

                    text =
                        clean_text(
                            block.inner_text()
                        )

                    if not text:
                        continue

                    # ----------------------------------------
                    # Rating
                    # ----------------------------------------

                    rating = None

                    try:

                        rating_elements =
                            block.locator(
                                "div._3LWZlK"
                            )

                        if (
                            rating_elements.count()
                            > 0
                        ):

                            rating =
                                parse_rating(
                                    rating_elements
                                    .first
                                    .inner_text()
                                )

                    except Exception:
                        pass

                    if rating is None:

                        rating =
                            parse_rating(
                                text
                            )

                    # ----------------------------------------
                    # Verified Buyer
                    # ----------------------------------------

                    verified_purchase =
                        "Verified Buyer"
                        in text

                    # ----------------------------------------
                    # Date
                    # ----------------------------------------

                    date = ""

                    date_match = re.search(
                        r"(yesterday|"
                        r"\d+\s+days?\s+ago|"
                        r"\d+\s+weeks?\s+ago|"
                        r"\d+\s+months?\s+ago|"
                        r"\d+\s+years?\s+ago)",
                        text,
                        re.I
                    )

                    if date_match:

                        date =
                            date_match.group(1)

                    # ----------------------------------------
                    # Helpful counts
                    # ----------------------------------------

                    helpful_count = 0
                    not_helpful_count = 0

                    try:

                        nums =
                            re.findall(
                                r"(?<!\d)\d+(?!\d)",
                                text
                            )

                        # Flipkart's displayed helpful
                        # values usually appear near the
                        # bottom of the review.

                        if len(nums) >= 2:

                            helpful_count =
                                int(nums[-2])

                            not_helpful_count =
                                int(nums[-1])

                    except Exception:
                        pass

                    # ----------------------------------------
                    # Review text
                    # ----------------------------------------

                    review_text = ""

                    # Common review body selectors
                    review_text_selectors = [

                        "div.t-ZTKy",

                        "div[class*='t-ZTKy']",

                        "div[class*='reviewText']",

                        "div[class*='ReviewText']"

                    ]

                    for selector in (
                        review_text_selectors
                    ):

                        try:

                            loc =
                                block.locator(
                                    selector
                                )

                            if loc.count() > 0:

                                review_text =
                                    clean_text(
                                        loc.first
                                        .inner_text()
                                    )

                                if review_text:
                                    break

                        except Exception:
                            continue

                    # ----------------------------------------
                    # Fallback text extraction
                    # ----------------------------------------

                    if not review_text:

                        lines =
                            [
                                x.strip()
                                for x in
                                text.splitlines()
                                if x.strip()
                            ]

                        skip = {

                            "Verified Buyer",

                            "Flipkart Customer",

                            "Helpful",

                            "Report",

                            "Reply"

                        }

                        possible = []

                        for line in lines:

                            lower =
                                line.lower()

                            if line in skip:
                                continue

                            if re.fullmatch(
                                r"[1-5]",
                                line
                            ):
                                continue

                            if re.fullmatch(
                                r"\d+",
                                line
                            ):
                                continue

                            if (
                                "ago"
                                in lower
                            ):
                                continue

                            if (
                                len(line) >= 8
                            ):

                                possible.append(
                                    line
                                )

                        # The longer descriptive
                        # line is usually the review.
                        if possible:

                            review_text =
                                max(
                                    possible,
                                    key=len
                                )

                    # ----------------------------------------
                    # Review title
                    # ----------------------------------------

                    review_title = ""

                    title_candidates = [

                        "p._2-N8zT",

                        "p[class*='_2-N8zT']",

                        "div[class*='title']"

                    ]

                    for selector in (
                        title_candidates
                    ):

                        try:

                            loc =
                                block.locator(
                                    selector
                                )

                            if loc.count() > 0:

                                review_title =
                                    clean_text(
                                        loc.first
                                        .inner_text()
                                    )

                                if review_title:
                                    break

                        except Exception:
                            continue

                    # ----------------------------------------
                    # Final review text
                    # ----------------------------------------

                    if (
                        review_title
                        and review_text
                    ):

                        final_text =
                            review_text

                    elif review_text:

                        final_text =
                            review_text

                    else:

                        final_text =
                            review_title

                    if not final_text:
                        continue

                    review = {

                        "review_number":
                            index,

                        "rating":
                            rating,

                        "title":
                            review_title,

                        "text":
                            final_text,

                        "date":
                            date,

                        "verified_purchase":
                            verified_purchase,

                        "helpful_count":
                            helpful_count,

                        "not_helpful_count":
                            not_helpful_count

                    }

                    reviews.append(
                        review
                    )

                    print(
                        f"Review {index}:",
                        review
                    )

                except Exception as e:

                    print(
                        f"Review {index} failed:",
                        e
                    )

            # ------------------------------------------------
            # FALLBACK PARSER
            # ------------------------------------------------

            if not reviews:

                print(
                    "DOM review selectors did not "
                    "produce usable reviews."
                )

                print(
                    "Attempting fallback parser..."
                )

                lines = [
                    x.strip()
                    for x in body_text.splitlines()
                    if x.strip()
                ]

                verified_indexes = [

                    i
                    for i, line
                    in enumerate(lines)
                    if line == "Verified Buyer"

                ]

                for idx, vi in enumerate(
                    verified_indexes[:10],
                    start=1
                ):

                    try:

                        section_start =
                            max(
                                0,
                                vi - 8
                            )

                        section_end =
                            min(
                                len(lines),
                                vi + 5
                            )

                        section =
                            lines[
                                section_start:
                                section_end
                            ]

                        section_text =
                            " ".join(section)

                        rating =
                            parse_rating(
                                section_text
                            )

                        possible = []

                        for line in section:

                            if (
                                line
                                in [
                                    "Verified Buyer",
                                    "Flipkart Customer"
                                ]
                            ):
                                continue

                            if re.fullmatch(
                                r"\d+",
                                line
                            ):
                                continue

                            if "ago" in line.lower():
                                continue

                            if len(line) >= 8:

                                possible.append(
                                    line
                                )

                        if possible:

                            review_text =
                                max(
                                    possible,
                                    key=len
                                )

                            reviews.append({

                                "review_number":
                                    idx,

                                "rating":
                                    rating,

                                "title":
                                    "",

                                "text":
                                    review_text,

                                "date":
                                    "",

                                "verified_purchase":
                                    True,

                                "helpful_count":
                                    0,

                                "not_helpful_count":
                                    0

                            })

                    except Exception:
                        continue

        finally:

            browser.close()

    print()
    print(
        "Total Flipkart reviews extracted:",
        len(reviews)
    )

    return reviews


# ============================================================
# FEATURES CUSTOMERS LOVED
# ============================================================

def extract_flipkart_topics(
    url: str
) -> Dict:

    print()
    print(
        "Extracting Flipkart customer topics..."
    )
    print()

    topics = {}

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 1200
            }
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(
                7000
            )

            # Scroll to reviews

            try:

                locator =
                    page.get_by_text(
                        "Features customers loved",
                        exact=False
                    )

                if locator.count() > 0:

                    locator.first.scroll_into_view_if_needed()

                    page.wait_for_timeout(
                        1000
                    )

            except Exception:
                pass

            body =
                page.locator(
                    "body"
                ).inner_text()

            topic_names = [

                "Colour",

                "Comfort",

                "Material Quality",

                "Light Weight",

                "True to Specs"

            ]

            for topic in topic_names:

                found =
                    topic.lower()
                    in body.lower()

                topics[topic] = {

                    "found":
                        found,

                    "summary":
                        (
                            f"Customers mention {topic}"
                            if found
                            else ""
                        )

                }

                print(
                    f"{topic}:",
                    "found"
                    if found
                    else "not found"
                )

        finally:

            browser.close()

    return topics


# ============================================================
# MAIN FLIPKART SCAN
# ============================================================

def scan_flipkart_product(
    url: str
) -> Dict:

    print()
    print("=" * 60)
    print("VERISIGHT FLIPKART SCANNER")
    print("=" * 60)

    print()
    print("URL:")
    print(url)

    print()
    print("[1/3] Extracting product information...")

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 1200
            }
        )

        try:

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(
                7000
            )

            product =
                extract_flipkart_product_info(
                    page
                )

        finally:

            browser.close()

    print(
        "Overall Rating:",
        product.get("overall_rating")
    )

    print(
        "Total Ratings:",
        product.get("total_ratings")
    )

    print()
    print(
        "[2/3] Extracting reviews..."
    )

    reviews =
        extract_flipkart_reviews(
            url
        )

    print()
    print(
        "[3/3] Extracting customer topics..."
    )

    topics =
        extract_flipkart_topics(
            url
        )

    return {

        "platform":
            "Flipkart",

        "product":
            product,

        "review_count":
            len(reviews),

        "reviews":
            reviews,

        "customers_say":
            topics

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_url = (
        "https://www.flipkart.com/"
        "red-tape-sneakers-men/"
        "p/itm7e392c192a605"
        "?pid=SHOHM9TGGHXYTTN4"
        "&lid=LSTSHOHM9TGGHXYTTN4GZBG42"
        "&marketplace=FLIPKART"
    )

    result =
        scan_flipkart_product(
            test_url
        )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        result
    )