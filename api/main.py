from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from api.schemas import (
    ReviewRequest,
    ReviewResponse
)

from api.verisight_engine import analyze_review


app = FastAPI(
    title="VeriSight API",
    description="AI-Generated and Spam Review Detection System",
    version="1.0.0"
)


BASE_DIR = Path(__file__).resolve().parent.parent

HTML_FILE = BASE_DIR / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def home():

    return HTML_FILE.read_text(
        encoding="utf-8"
    )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "stage1": "loaded",
        "stage2": "loaded"
    }


@app.post(
    "/api/v1/analyze-review",
    response_model=ReviewResponse
)
def analyze(request: ReviewRequest):

    try:

        result = analyze_review(
            review_text=request.review_text,
            rating=request.rating,
            mode=request.mode
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )