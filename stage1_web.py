from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "mlp_stage1_v2.keras"


# ==========================================================
# LOAD MODELS
# ==========================================================

print("=" * 60)
print("VERISIGHT - STAGE 1 AI REVIEW DETECTOR")
print("=" * 60)

print("\nLoading MiniLM...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Stage 1 MLP...")
model = load_model(MODEL_PATH)

print("Models loaded successfully!")


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="VeriSight Stage 1",
    description="AI Generated Review Detection"
)


# ==========================================================
# REQUEST MODEL
# ==========================================================

class ReviewRequest(BaseModel):
    review: str


# ==========================================================
# WEB PAGE
# ==========================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>VeriSight - AI Review Detector</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #24204f,
            #0a0b16 45%,
            #05060b
        );

    color: #ffffff;

    min-height: 100vh;

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 30px;
}


.container {

    width: 100%;

    max-width: 850px;

}


.header {

    text-align: center;

    margin-bottom: 30px;
}


.logo {

    font-size: 48px;

    font-weight: 900;

    letter-spacing: 3px;

}


.logo span {

    color: #8b5cf6;
}


.subtitle {

    margin-top: 8px;

    color: #94a3b8;

    font-size: 16px;

}


.status {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    margin-top: 15px;

    padding: 8px 14px;

    border-radius: 20px;

    background: rgba(34,197,94,0.10);

    border: 1px solid rgba(34,197,94,0.25);

    color: #86efac;

    font-size: 13px;
}


.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

}


.card {

    background:
        rgba(15, 23, 42, 0.90);

    border:

        1px solid

        rgba(148,163,184,0.15);

    border-radius: 18px;

    padding: 28px;

    box-shadow:

        0 25px 70px

        rgba(0,0,0,0.45);

    backdrop-filter: blur(15px);

}


.label {

    display: block;

    margin-bottom: 10px;

    color: #cbd5e1;

    font-weight: 700;

    font-size: 14px;
}


textarea {

    width: 100%;

    min-height: 190px;

    resize: vertical;

    border: 1px solid #334155;

    border-radius: 12px;

    background: #080d18;

    color: #f8fafc;

    padding: 16px;

    font-size: 15px;

    line-height: 1.6;

    outline: none;

}


textarea:focus {

    border-color: #8b5cf6;

    box-shadow:
        0 0 0 3px
        rgba(139,92,246,0.15);

}


button {

    width: 100%;

    margin-top: 18px;

    padding: 16px;

    border: none;

    border-radius: 12px;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );

    color: white;

    font-size: 16px;

    font-weight: 800;

    cursor: pointer;

    transition: 0.2s;
}


button:hover {

    transform: translateY(-1px);

    box-shadow:
        0 10px 30px
        rgba(139,92,246,0.30);
}


button:disabled {

    opacity: 0.6;

    cursor: wait;

    transform: none;
}


#result {

    margin-top: 25px;

    display: none;
}


.result-card {

    padding: 28px;

    border-radius: 16px;

    background: #0b1220;

    border: 1px solid #243047;

    text-align: center;
}


.result-icon {

    font-size: 45px;

    margin-bottom: 10px;
}


.result-label {

    color: #94a3b8;

    font-size: 13px;

    letter-spacing: 1px;

    font-weight: 700;
}


.prediction {

    font-size: 30px;

    font-weight: 900;

    margin: 10px 0 25px;
}


.ai-title {

    color: #a78bfa;
}


.human-title {

    color: #4ade80;
}


.probability-label {

    display: flex;

    justify-content: space-between;

    color: #cbd5e1;

    font-size: 14px;

    margin-bottom: 8px;
}


.progress {

    height: 14px;

    width: 100%;

    background: #1e293b;

    border-radius: 20px;

    overflow: hidden;
}


.progress-bar {

    height: 100%;

    border-radius: 20px;

    transition: width 0.7s ease;
}


.ai-bar {

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #a855f7
        );
}


.human-bar {

    background:
        linear-gradient(
            90deg,
            #22c55e,
            #4ade80
        );
}


.note {

    margin-top: 20px;

    color: #64748b;

    font-size: 12px;

    line-height: 1.5;
}


.error {

    color: #fca5a5;

    background: #35151a;

    border: 1px solid #7f1d1d;

    padding: 15px;

    border-radius: 10px;

    margin-top: 20px;
}


.footer {

    text-align: center;

    color: #475569;

    font-size: 12px;

    margin-top: 20px;
}


@media(max-width:600px) {

    body {
        padding: 15px;
    }

    .logo {
        font-size: 36px;
    }

    .card {
        padding: 20px;
    }

}

</style>

</head>


<body>

<div class="container">


    <!-- HEADER -->

    <div class="header">

        <div class="logo">
            Veri<span>Sight</span>
        </div>

        <div class="subtitle">
            AI-Generated Review Detection
        </div>

        <div class="status">

            <div class="status-dot"></div>

            Stage 1 Detection Engine Online

        </div>

    </div>


    <!-- INPUT CARD -->

    <div class="card">

        <label class="label">
            Enter Review
        </label>

        <textarea
            id="review"
            placeholder="Paste or type the review you want to analyze..."
        ></textarea>


        <button
            id="analyzeButton"
            onclick="analyzeReview()"
        >
            🔍 ANALYZE REVIEW
        </button>


        <div id="result"></div>

    </div>


    <div class="footer">

        VeriSight • Stage 1 AI Review Detection

    </div>


</div>


<script>

async function analyzeReview() {

    const review =
        document.getElementById("review").value.trim();

    const button =
        document.getElementById("analyzeButton");

    const result =
        document.getElementById("result");


    if (!review) {

        result.style.display = "block";

        result.innerHTML = `
            <div class="error">
                Please enter a review first.
            </div>
        `;

        return;
    }


    button.disabled = true;

    button.innerText = "⏳ ANALYZING...";


    result.style.display = "block";

    result.innerHTML = `
        <div class="result-card">

            <div class="result-icon">
                🔄
            </div>

            <div class="result-label">
                VERISIGHT
            </div>

            <div class="prediction">
                Analyzing Review...
            </div>

        </div>
    `;


    try {

        const response = await fetch(
            "/predict",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    review: review
                })

            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed"
            );

        }


        const aiProbability =
            data.ai_probability;

        const prediction =
            data.prediction;


        const isAI =
            prediction === "AI Generated";


        const icon =
            isAI ? "🤖" : "👤";


        const predictionClass =
            isAI
                ? "ai-title"
                : "human-title";


        const barClass =
            isAI
                ? "ai-bar"
                : "human-bar";


        result.innerHTML = `

            <div class="result-card">

                <div class="result-icon">
                    ${icon}
                </div>

                <div class="result-label">
                    PREDICTION
                </div>

                <div class="
                    prediction
                    ${predictionClass}
                ">
                    ${prediction}
                </div>


                <div class="probability-label">

                    <span>
                        AI Probability
                    </span>

                    <strong>
                        ${aiProbability.toFixed(2)}%
                    </strong>

                </div>


                <div class="progress">

                    <div
                        class="
                            progress-bar
                            ${barClass}
                        "
                        style="
                            width:${aiProbability}%;
                        "
                    ></div>

                </div>


                <div class="note">

                    Probability produced by the
                    VeriSight Stage 1 MLP model
                    using a MiniLM text embedding.

                </div>

            </div>

        `;

    }


    catch(error) {

        result.innerHTML = `

            <div class="error">

                ❌ ${error.message}

            </div>

        `;

    }


    finally {

        button.disabled = false;

        button.innerText =
            "🔍 ANALYZE REVIEW";

    }

}

</script>


</body>

</html>
"""


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML_PAGE


@app.post("/predict")
def predict(request: ReviewRequest):

    review = request.review.strip()

    if not review:

        return {
            "error": "Review cannot be empty."
        }


    # ------------------------------------------------------
    # MiniLM embedding
    # ------------------------------------------------------

    embedding = encoder.encode(
        [review],
        convert_to_numpy=True
    )


    # ------------------------------------------------------
    # Stage 1 MLP prediction
    # ------------------------------------------------------

    probability = model.predict(
        embedding,
        verbose=0
    )[0][0]


    ai_probability = float(probability * 100)


    # ------------------------------------------------------
    # Classification
    # ------------------------------------------------------

    if probability >= 0.5:

        prediction = "AI Generated"

    else:

        prediction = "Human Written"


    return {

        "prediction": prediction,

        "ai_probability":
            round(ai_probability, 2)

    }


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8010
    )