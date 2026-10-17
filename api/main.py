from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.verisight_engine import analyze_amazon_reviews
from api.amazon_scanner import (
    extract_amazon_reviews,
    extract_customers_say_topics,
    extract_amazon_product_info,
)
from api.universal_scanner import scan_page

from playwright.sync_api import sync_playwright


app = FastAPI(
    title="VeriSight API",
    version="3.0.0",
    description="Universal review integrity scanner with Amazon-specific scanning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str


def is_amazon_url(url: str) -> bool:
    url_lower = (url or "").lower()

    amazon_domains = [
        "amazon.in",
        "amazon.com",
        "amazon.co.uk",
        "amazon.ae",
        "amazon.de",
        "amazon.fr",
        "amazon.it",
        "amazon.es",
        "amazon.ca",
        "amazon.com.au",
        "amazon.co.jp",
    ]

    return any(domain in url_lower for domain in amazon_domains)


def analyze_amazon(url: str):
    """
    Keep the existing Amazon-specific scanner.

    Amazon gets:
    - Amazon review extraction
    - Amazon product information
    - Customers Say topic interaction
    - Stage 1 + Stage 2 analysis
    """

    try:
        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(headless=True)

            page = browser.new_page(
                viewport={
                    "width": 1440,
                    "height": 1000,
                }
            )

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000,
            )

            page.wait_for_timeout(3000)

            product = extract_amazon_product_info(page)

            browser.close()

        reviews = extract_amazon_reviews(url)

        customers_say = extract_customers_say_topics(url)

        if not reviews:
            return {
                "success": False,
                "error": "No Amazon reviews could be extracted.",
                "platform": "amazon",
                "url": url,
                "product": product,
                "review_count": 0,
                "reviews": [],
                "summary": {},
                "customers_say": customers_say,
            }

        analysis = analyze_amazon_reviews(reviews)

        return {
            "success": True,
            "platform": "amazon",
            "scanner": "amazon_specific",
            "url": url,
            "product": product,
            "review_count": len(reviews),
            "reviews": analysis.get("reviews", []),
            "summary": analysis.get("summary", {}),
            "customers_say": customers_say,
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
            "platform": "amazon",
            "scanner": "amazon_specific",
            "url": url,
            "product": {},
            "review_count": 0,
            "reviews": [],
            "summary": {},
            "customers_say": {},
        }


def analyze_universal(url: str):
    """
    Universal scanner for every non-Amazon platform.

    The scanner normalizes reviews into the format expected
    by the VeriSight ML engine.
    """

    try:
        scan_result = scan_page(
            url,
            review_limit=20,
            headless=True,
        )

        if not scan_result.get("success"):
            return {
                "success": False,
                "error": scan_result.get(
                    "error",
                    "Universal scanner failed.",
                ),
                "platform": scan_result.get(
                    "platform",
                    "unknown",
                ),
                "scanner": "universal",
                "url": url,
                "product": scan_result.get(
                    "product",
                    {},
                ),
                "review_count": 0,
                "reviews": [],
                "summary": {},
                "customers_say": scan_result.get(
                    "customers_say",
                    {},
                ),
            }

        reviews = scan_result.get(
            "reviews",
            [],
        )

        if not reviews:
            return {
                "success": False,
                "error": "No reviews found on this page.",
                "platform": scan_result.get(
                    "platform",
                    "unknown",
                ),
                "scanner": "universal",
                "url": url,
                "product": scan_result.get(
                    "product",
                    {},
                ),
                "review_count": 0,
                "reviews": [],
                "summary": {},
                "customers_say": scan_result.get(
                    "customers_say",
                    {},
                ),
            }

        # analyze_amazon_reviews expects the normalized review
        # structure, which the universal scanner already provides.
        analysis = analyze_amazon_reviews(reviews)

        return {
            "success": True,
            "platform": scan_result.get(
                "platform",
                "unknown",
            ),
            "scanner": "universal",
            "url": url,
            "product": scan_result.get(
                "product",
                {},
            ),
            "review_count": len(reviews),
            "reviews": analysis.get(
                "reviews",
                [],
            ),
            "summary": analysis.get(
                "summary",
                {},
            ),
            "customers_say": scan_result.get(
                "customers_say",
                {},
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
            "platform": "unknown",
            "scanner": "universal",
            "url": url,
            "product": {},
            "review_count": 0,
            "reviews": [],
            "summary": {},
            "customers_say": {},
        }


@app.get("/", response_class=HTMLResponse)
def root():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VeriSight</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <h1>VeriSight</h1>
        <p>Universal Review Integrity Scanner API</p>
        <p>
            Amazon uses the Amazon-specific scanner.
            Other platforms use the Universal Scanner.
        </p>
    </body>
    </html>
    """


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "stage1": "loaded",
        "stage2": "page_model_loaded",
        "amazon_scanner": "loaded",
        "universal_scanner": "loaded",
        "customers_say": "loaded",
        "product_info": "loaded",
    }


@app.post("/api/v1/analyze-amazon")
def analyze_amazon_endpoint(request: AnalyzeRequest):

    if not request.url:
        raise HTTPException(
            status_code=400,
            detail="URL is required.",
        )

    # Keep this endpoint for backward compatibility with
    # the existing Chrome extension.
    #
    # Amazon -> Amazon-specific scanner
    # Other platforms -> Universal Scanner
    if is_amazon_url(request.url):
        return analyze_amazon(request.url)

    return analyze_universal(request.url)


@app.post("/api/v1/analyze")
def analyze_endpoint(request: AnalyzeRequest):

    if not request.url:
        raise HTTPException(
            status_code=400,
            detail="URL is required.",
        )

    if is_amazon_url(request.url):
        return analyze_amazon(request.url)

    return analyze_universal(request.url)


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
