from pydantic import BaseModel, Field
from typing import Optional


class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=5)
    rating: Optional[float] = Field(default=None, ge=1, le=5)

    # For demo mode, this identifies a row from stage2_test_hybrid.csv
    test_row_index: Optional[int] = Field(default=None, ge=0)


class Signals(BaseModel):
    ai_detection: str
    spam_detection: str
    rating_text_consistency: str


class ReviewResponse(BaseModel):
    ai_probability: float
    human_probability: float

    spam_probability: float
    genuine_probability: float

    classification: str
    risk_level: str

    signals: Signals