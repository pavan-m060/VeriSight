import re
from typing import Any, Dict, List, Optional
from playwright.sync_api import sync_playwright

SUPPORTED_DOMAINS = {
    "amazon": ["amazon.in", "amazon.com", "amazon.co.uk", "amazon.ae"],
    "flipkart": ["flipkart.com"],
    "myntra": ["myntra.com"],
    "meesho": ["meesho.com"],
    "shopsy": ["shopsy.in"],
    "ajio": ["ajio.com"],
    "nykaa": ["nykaa.com", "nykaafashion.com"],
    "zepto": ["zeptonow.com", "zepto.com"],
    "instamart": ["instamart.in", "swiggy.com"],
}

FLIPKART_REVIEW_TITLES = [
    "Perfect product!", "Delightful", "Awesome", "Terrific purchase",
    "Mind-blowing purchase", "Simply awesome", "Must buy!", "Worthless",
    "Just wow!", "Wonderful", "Excellent", "Good quality product",
]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def detect_platform(url: str) -> str:
    u = (url or "").lower()
    for platform, domains in SUPPORTED_DOMAINS.items():
        if any(domain in u for domain in domains):
            return platform
    return "unknown"


def parse_rating(text: str) -> Optional[float]:
    text = clean_text(text)
    for pattern in (
        r"\b([0-5](?:\.[0-9])?)\s*(?:out of\s*5|/5)\b",
        r"\b([0-5](?:\.[0-9])?)\s*★",
        r"^\s*([0-5](?:\.[0-9])?)\s*$",
    ):
        m = re.search(pattern, text, re.I)
        if m:
            try:
                value = float(m.group(1))
                if 0 <= value <= 5:
                    return value
            except ValueError:
                pass
    return None


def extract_date(text: str) -> str:
    text = clean_text(text)
    patterns = [
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4}\b",
        r"\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b",
        r"\b(?:today|yesterday)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(0)
    return ""


def parse_helpful_count(text: str) -> int:
    text = clean_text(text).lower()
    for pattern in (
        r"(\d+)\s*(?:people\s*)?found\s*this\s*(?:helpful|useful)",
        r"(\d+)\s*(?:helpful|useful)",
        r"helpful\s*[:\-]?\s*(\d+)",
    ):
        m = re.search(pattern, text, re.I)
        if m:
            return int(m.group(1))
    return 0


def extract_product_information(page, platform: str) -> Dict[str, Any]:
    body = clean_text(page.locator("body").inner_text(timeout=5000))
    rating = None
    total = 0

    # Flipkart currently exposes the product summary as: 4.3 | 650
    if platform == "flipkart":
        m = re.search(r"\b([0-5](?:\.[0-9]))\s*\|\s*([\d,]+)\b", body)
        if m:
            rating = float(m.group(1))
            total = int(m.group(2).replace(",", ""))

    rating_selectors = [
        "#acrPopover",
        "[data-hook='average-star-rating']",
        "[data-hook='rating-out-of-text']",
        ".a-icon-alt",
        "[class*='rating']",
    ]
    if rating is None:
        for selector in rating_selectors:
            try:
                loc = page.locator(selector)
                for i in range(min(loc.count(), 10)):
                    e = loc.nth(i)
                    rating = parse_rating(e.inner_text(timeout=800))
                    if rating is None:
                        rating = parse_rating(e.get_attribute("title") or "")
                    if rating is not None:
                        break
            except Exception:
                continue
            if rating is not None:
                break

    if total == 0:
        for selector in [
            "#acrCustomerReviewText",
            "[data-hook='total-review-count']",
            "[class*='review-count']",
        ]:
            try:
                loc = page.locator(selector)
                for i in range(min(loc.count(), 10)):
                    nums = re.findall(r"[\d,]+", clean_text(loc.nth(i).inner_text(timeout=800)))
                    if nums:
                        total = int(nums[-1].replace(",", ""))
                        break
            except Exception:
                continue
            if total:
                break

    if total == 0:
        m = re.search(r"([\d,]+)\s+(?:ratings|rating|reviews)", body, re.I)
        if m:
            total = int(m.group(1).replace(",", ""))

    return {"overall_rating": rating, "total_ratings": total}


def _flipkart_review_blocks(page, limit: int) -> List[Any]:
    """Find Flipkart review cards without depending on one fixed CSS class.

    Flipkart changes its generated class names frequently.  Instead of relying
    on a hard-coded review-title list, use the stable semantic signals that
    appear inside a review: a 1-5 rating, a date/relative date, and usually
    "Verified Buyer".  For each such node we walk up a few ancestors and keep
    the smallest ancestor that looks like a complete review card.
    """
    candidates = []
    seen = set()

    try:
        verified_nodes = page.get_by_text(re.compile(r"Verified Buyer|Verified Purchase", re.I))
        count = min(verified_nodes.count(), 100)

        for i in range(count):
            node = verified_nodes.nth(i)
            try:
                # Walk upward and select the first compact ancestor that has
                # the signals of a review card.
                for level in range(1, 8):
                    card = node.locator("xpath=" + "/.." * level)
                    text = clean_text(card.inner_text(timeout=500))

                    if not (30 <= len(text) <= 1800):
                        continue
                    if not re.search(r"Verified Buyer|Verified Purchase", text, re.I):
                        continue
                    if not extract_date(text):
                        continue
                    if not re.search(r"(?:^|\n|\s)[1-5](?:\.0)?(?:\s|$)", text):
                        # Some cards expose the rating as 4.3 rather than 4.
                        if parse_rating(text) is None:
                            continue

                    key = text[:700]
                    if key not in seen:
                        seen.add(key)
                        candidates.append(card)
                    break
            except Exception:
                continue

            if len(candidates) >= limit:
                break
    except Exception:
        pass

    # Fallback: search the Ratings and reviews area for compact containers
    # containing the same semantic signals.
    if len(candidates) < limit:
        try:
            body_loc = page.locator("body")
            all_divs = body_loc.locator("div")
            for i in range(min(all_divs.count(), 3000)):
                e = all_divs.nth(i)
                try:
                    text = clean_text(e.inner_text(timeout=150))
                    if not (40 <= len(text) <= 1200):
                        continue
                    if not re.search(r"Verified Buyer|Verified Purchase", text, re.I):
                        continue
                    if not extract_date(text) or parse_rating(text) is None:
                        continue

                    key = text[:700]
                    if key not in seen:
                        seen.add(key)
                        candidates.append(e)
                        if len(candidates) >= limit:
                            break
                except Exception:
                    continue
        except Exception:
            pass

    # Final fallback: inspect reasonably-sized divs for a rating + date.
    # This is intentionally broad, but excludes huge page containers.
    if len(candidates) < limit:
        try:
            divs = page.locator("div")
            for i in range(min(divs.count(), 5000)):
                e = divs.nth(i)
                text = clean_text(e.inner_text(timeout=100))
                if not (35 <= len(text) <= 900):
                    continue
                if not extract_date(text):
                    continue
                if parse_rating(text) is None:
                    continue

                # Avoid product summary/header containers.
                if re.search(
                    r"add to cart|buy now|delivery|select size|seller|₹|offers",
                    text,
                    re.I,
                ):
                    continue

                key = text[:700]
                if key not in seen:
                    seen.add(key)
                    candidates.append(e)
                    if len(candidates) >= limit:
                        break
        except Exception:
            pass

    return candidates[:limit]


def _flipkart_review_from_card(card, number: int) -> Dict[str, Any]:
    full = clean_text(card.inner_text(timeout=1000))
    rating = parse_rating(full)
    date = extract_date(full)
    verified = bool(re.search(r"\bVerified Buyer\b|\bVerified Purchase\b", full, re.I))

    lines = [clean_text(x) for x in card.inner_text(timeout=1000).splitlines() if clean_text(x)]

    # Remove obvious metadata before looking for title/body text.
    metadata_patterns = [
        r"^[1-5](?:\.0)?$",
        r"^\d+(?:\.\d+)?\s*(?:out of\s*5|/5)$",
        r"^Verified Buyer$",
        r"^Verified Purchase$",
        r"^\d+\s*(?:people\s*)?found this (?:helpful|useful)$",
        r"^Helpful$",
        r"^Share$",
    ]

    content_lines = []
    for line in lines:
        if date and line.lower() == date.lower():
            continue
        if any(re.fullmatch(pattern, line, re.I) for pattern in metadata_patterns):
            continue
        if re.fullmatch(r"[1-5](?:\.0)?", line):
            continue
        content_lines.append(line)

    title = ""
    if content_lines:
        # The first short line is normally the review title.
        for line in content_lines:
            if 2 <= len(line) <= 140:
                title = line
                break

    review_text = ""

    # First try common review-body selectors, scoped to this card.
    body_selectors = [
        "[class*='v1zwn27']",
        "[class*='review-text']",
        "[class*='reviewText']",
        "[class*='review-body']",
        "[class*='reviewBody']",
        "p",
    ]

    for selector in body_selectors:
        try:
            loc = card.locator(selector)
            vals = []
            for i in range(min(loc.count(), 30)):
                value = clean_text(loc.nth(i).inner_text(timeout=400))
                if 3 <= len(value) <= 1000:
                    vals.append(value)
            if vals:
                filtered = [
                    v for v in vals
                    if v.lower() not in {
                        title.lower(),
                        date.lower(),
                        "verified buyer",
                        "verified purchase",
                    }
                ]
                if filtered:
                    review_text = max(filtered, key=len)
                    break
        except Exception:
            continue

    # Final fallback: choose the longest content line that is not metadata.
    if not review_text:
        usable = [
            x for x in content_lines
            if x.lower() != title.lower()
            and len(x) >= 3
        ]
        if usable:
            review_text = max(usable, key=len)
        elif title:
            review_text = title

    return {
        "review_number": number,
        "rating": rating,
        "text": review_text,
        "review_text": review_text,
        "title": title,
        "date": date,
        "reviewer": "",
        "verified_purchase": verified,
        "helpful_count": parse_helpful_count(full),
    }


def _generic_review_blocks(page, limit: int) -> List[Any]:
    selectors = [
        "div[data-hook='review']",
        "[data-review-id]",
        "[class*='review-card']",
        "[class*='reviewCard']",
        "[class*='review-item']",
        "[class*='reviewItem']",
    ]
    for selector in selectors:
        try:
            loc = page.locator(selector)
            blocks = []
            seen = set()
            for i in range(min(loc.count(), 100)):
                e = loc.nth(i)
                text = clean_text(e.inner_text(timeout=800))
                if len(text) < 20 or text in seen:
                    continue
                if parse_rating(text) is not None and (extract_date(text) or re.search(r"verified buyer|verified purchase", text, re.I)):
                    seen.add(text)
                    blocks.append(e)
            if blocks:
                return blocks[:limit]
        except Exception:
            continue
    return []


def _generic_review(block, number: int) -> Dict[str, Any]:
    full = clean_text(block.inner_text(timeout=1000))
    review_text = full
    for selector in [
        "[data-hook='review-body']",
        "[class*='review-text']",
        "[class*='reviewText']",
        "[class*='review-body']",
        "[class*='reviewBody']",
        "p",
    ]:
        try:
            loc = block.locator(selector)
            vals = [clean_text(loc.nth(i).inner_text(timeout=500)) for i in range(min(loc.count(), 10))]
            vals = [v for v in vals if len(v) >= 10]
            if vals:
                review_text = max(vals, key=len)
                break
        except Exception:
            pass
    return {
        "review_number": number,
        "rating": parse_rating(full),
        "text": review_text,
        "review_text": review_text,
        "title": "",
        "date": extract_date(full),
        "reviewer": "",
        "verified_purchase": bool(re.search(r"verified buyer|verified purchase", full, re.I)),
        "helpful_count": parse_helpful_count(full),
    }



def _flipkart_reviews_from_page_text(page, limit: int = 20) -> List[Dict[str, Any]]:
    """Parse the currently rendered Flipkart review sequence from page text."""
    try:
        raw = page.locator("body").inner_text(timeout=10000)
    except Exception:
        return []

    lines = [clean_text(x) for x in raw.splitlines() if clean_text(x)]
    titles = set(FLIPKART_REVIEW_TITLES)
    reviews = []
    seen = set()

    for i, line in enumerate(lines):
        if line not in titles or i + 3 >= len(lines):
            continue

        date = extract_date(lines[i + 1])
        if not date:
            continue

        body = lines[i + 2]
        reviewer = lines[i + 3]

        verified_index = None
        for j in range(i + 4, min(i + 8, len(lines))):
            if re.search(r"\bVerified Buyer\b|\bVerified Purchase\b", lines[j], re.I):
                verified_index = j
                break

        if verified_index is None:
            continue

        rating = None
        if i > 0:
            m = re.fullmatch(r"([1-5])(?:\.0)?", lines[i - 1])
            if m:
                rating = float(m.group(1))

        if rating is None:
            continue

        key = (rating, date.lower(), body.lower(), reviewer.lower())
        if key in seen:
            continue
        seen.add(key)

        helpful = 0
        for j in range(verified_index + 1, min(verified_index + 4, len(lines))):
            m = re.fullmatch(r"\d+", lines[j])
            if m:
                helpful = int(m.group(0))
                break

        reviews.append({
            "review_number": len(reviews) + 1,
            "rating": rating,
            "text": body,
            "review_text": body,
            "title": line,
            "date": date,
            "reviewer": reviewer,
            "verified_purchase": True,
            "helpful_count": helpful,
        })

        if len(reviews) >= limit:
            break

    return reviews


def extract_reviews(page, platform: str, limit: int = 20) -> List[Dict[str, Any]]:
    if platform == "flipkart":
        reviews = _flipkart_reviews_from_page_text(page, limit)
        if reviews:
            return reviews

        blocks = _flipkart_review_blocks(page, limit)
        return [_flipkart_review_from_card(b, i) for i, b in enumerate(blocks, 1)]

    blocks = _generic_review_blocks(page, limit)
    return [_generic_review(b, i) for i, b in enumerate(blocks, 1)]



def extract_customer_topics(page) -> Dict[str, Dict[str, Any]]:
    """Extract only topics inside Flipkart's 'Features customers loved' section."""
    try:
        body = page.locator("body").inner_text(timeout=5000)
    except Exception:
        return {}

    lines = [clean_text(x) for x in body.splitlines() if clean_text(x)]

    try:
        start = next(
            i for i, x in enumerate(lines)
            if re.fullmatch(r"Features customers loved", x, re.I)
        )
    except StopIteration:
        return {}

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.fullmatch(r"Ratings and reviews", lines[i], re.I):
            end = i
            break

    section = lines[start + 1:end]

    known_topics = [
        "Quality", "Comfort", "Value for money", "Appearance",
        "Material Quality", "Light Weight", "True to Specs", "Colour",
        "Color", "Size", "Fit", "Design"
    ]

    topics = {}
    for topic in known_topics:
        if any(x.lower() == topic.lower() for x in section):
            topics[topic] = {
                "found": True,
                "clicked": False,
                "summary": topic,
                "positive": None,
                "negative": None,
            }

    return topics


def scan_page(url: str, review_limit: int = 20, headless: bool = True) -> Dict[str, Any]:
    platform = detect_platform(url)
    if not url or not re.match(r"^https?://", url, re.I):
        return {
            "success": False, "error": "Invalid URL", "platform": platform,
            "url": url, "product": {}, "review_count": 0,
            "reviews": [], "customers_say": {},
        }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page(
                viewport={"width": 1440, "height": 1000},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/154.0.0.0 Safari/537.36"
                ),
            )
            try:
                page.goto(
                    url,
                    wait_until="commit",
                    timeout=60000
                )
            except Exception as first_error:
                print("Normal navigation failed:", first_error)
            try:
                page.goto(
                    url,
                    wait_until="load",
                    timeout=60000
                )
            except Exception as second_error:
                print("Fallback navigation failed:", second_error)
                raise second_error
            page.wait_for_timeout(8000)
            page.wait_for_timeout(4000)

            # Give lazy-loaded review components a chance to render.
            try:
                page.mouse.wheel(0, 2200)
                page.wait_for_timeout(1500)
                page.mouse.wheel(0, 2200)
                page.wait_for_timeout(1500)
            except Exception:
                pass

            product = extract_product_information(page, platform)
            reviews = extract_reviews(page, platform, review_limit)
            topics = extract_customer_topics(page)
            browser.close()

            return {
                "success": True,
                "platform": platform,
                "url": url,
                "product": product,
                "review_count": len(reviews),
                "reviews": reviews,
                "customers_say": topics,
            }
    except Exception as exc:
        return {
            "success": False, "error": str(exc), "platform": platform,
            "url": url, "product": {}, "review_count": 0,
            "reviews": [], "customers_say": {},
        }


if __name__ == "__main__":
    test_url = "https://www.flipkart.com/red-tape-sneakers-men/p/itm7e392c192a605?pid=SHOHM9TGGHXYTTN4&marketplace=FLIPKART"
    result = scan_page(test_url, review_limit=10)
    print("SUCCESS:", result["success"])
    print("PLATFORM:", result["platform"])
    print("PRODUCT:", result["product"])
    print("REVIEWS:", result["review_count"])
    for review in result["reviews"]:
        print(review)
