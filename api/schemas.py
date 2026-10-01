from pydantic import BaseModel, Field
from typing import List


class ReviewRequest(BaseModel):

    review_text: str = Field(
        ...,
        min_length=1
    )

    rating: float = 5.0

    # Stage 2 demo mode
    # genuine = Genuine
    # spam = Spam
    mode: str = "genuine"


class DetectionSignals(BaseModel):

    ai_signal: str

    spam_signal: str

    additional_signals: List[str] = []


class ReviewResponse(BaseModel):

    ai_probability: float

    human_probability: float

    spam_probability: float

    genuine_probability: float

    classification: str

    risk_level: str

    signals: DetectionSignals