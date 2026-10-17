document.addEventListener("DOMContentLoaded", () => {
    const scanButton = document.getElementById("scanButton");
    const status = document.getElementById("status");
    const results = document.getElementById("results");

    scanButton.addEventListener("click", scanCurrentPage);

    async function scanCurrentPage() {
        results.innerHTML = "";
        status.className = "status scanning";
        status.innerHTML = "🔍 Scanning current page...";
        scanButton.disabled = true;

        try {
            const tabs = await chrome.tabs.query({
                active: true,
                currentWindow: true
            });

            if (!tabs || !tabs[0] || !tabs[0].url) {
                throw new Error("Could not get the current page URL.");
            }

            const url = tabs[0].url;

            // Don't scan browser internal pages
            if (
                url.startsWith("chrome://") ||
                url.startsWith("edge://") ||
                url.startsWith("about:") ||
                url.startsWith("chrome-extension://")
            ) {
                throw new Error("This type of page cannot be scanned.");
            }

            status.innerHTML = "🌐 Opening page with VeriSight...";

            const response = await fetch(
                "http://127.0.0.1:8000/api/v1/analyze",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        url: url
                    })
                }
            );

            if (!response.ok) {
                let message = `Server error: ${response.status}`;

                try {
                    const errorData = await response.json();

                    if (errorData.detail) {
                        message = errorData.detail;
                    }
                } catch (_) {}

                throw new Error(message);
            }

            const data = await response.json();
            console.log("VERISIGHT API RESPONSE:", data);

            if (!data.success) {
                throw new Error(
                    data.error ||
                    data.message ||
                    "VeriSight could not analyze this page."
                );
            }

            status.className = "status success";
            status.innerHTML = "✓ Scan completed";

            displayResults(data);

        } catch (error) {
            console.error("VeriSight error:", error);

            status.className = "status error";
            status.innerHTML = "✕ Scan failed";

            results.innerHTML = `
                <div class="error-box">
                    <strong>Unable to scan this page</strong>
                    <p>${escapeHtml(error.message)}</p>
                    <small>
                        Make sure the VeriSight backend is running on
                        http://127.0.0.1:8000
                    </small>
                </div>
            `;
        } finally {
            scanButton.disabled = false;
        }
    }


    function displayResults(data) {

        const product = data.product || {};
        const summary = data.summary || {};
        const reviews = data.reviews || [];
        const customersSay = data.customers_say || {};

        const totalReviews =
            summary.total_reviews || reviews.length || 0;

        const humanReviews =
            summary.human_reviews || 0;

        const aiReviews =
            summary.ai_reviews || 0;

        const genuineReviews =
            summary.genuine_reviews || 0;

        const spamReviews =
            summary.spam_reviews || 0;


        // Platform
        const platform =
            data.platform ||
            data.scanner ||
            "unknown";


        // ---------------------------------------------------------
        // PRODUCT INFORMATION
        // ---------------------------------------------------------

        let html = `
            <div class="section">
                <div class="section-title">
                    🛍️ PRODUCT INFORMATION
                </div>

                <div class="product-card">

                    <div class="product-platform">
                        ${escapeHtml(
                            platform.toString().toUpperCase()
                        )}
                    </div>

                    ${
                        product.overall_rating !== undefined
                        ? `
                        <div class="product-rating">
                            ⭐ ${escapeHtml(
                                String(product.overall_rating)
                            )}
                        </div>
                        `
                        : ""
                    }

                    ${
                        product.total_ratings !== undefined
                        ? `
                        <div class="product-ratings-count">
                            ${escapeHtml(
                                String(product.total_ratings)
                            )} ratings
                        </div>
                        `
                        : ""
                    }

                </div>
            </div>
        `;


        // ---------------------------------------------------------
        // SUMMARY
        // ---------------------------------------------------------

        html += `
            <div class="section">

                <div class="section-title">
                    📊 VERISIGHT SUMMARY
                </div>

                <div class="summary-grid">

                    <div class="summary-card">
                        <div class="summary-number">
                            ${totalReviews}
                        </div>
                        <div class="summary-label">
                            Reviews
                        </div>
                    </div>

                    <div class="summary-card">
                        <div class="summary-number">
                            ${humanReviews}
                        </div>
                        <div class="summary-label">
                            Human
                        </div>
                    </div>

                    <div class="summary-card">
                        <div class="summary-number">
                            ${aiReviews}
                        </div>
                        <div class="summary-label">
                            AI
                        </div>
                    </div>

                    <div class="summary-card">
                        <div class="summary-number">
                            ${genuineReviews}
                        </div>
                        <div class="summary-label">
                            Genuine
                        </div>
                    </div>

                    <div class="summary-card">
                        <div class="summary-number">
                            ${spamReviews}
                        </div>
                        <div class="summary-label">
                            Potential Spam
                        </div>
                    </div>

                </div>

            </div>
        `;


        // ---------------------------------------------------------
        // PURCHASE INSIGHT
        // ---------------------------------------------------------

        if (totalReviews > 0) {

            const insight =
                calculatePurchaseInsight(
                    product,
                    summary
                );

            html += `
                <div class="section">

                    <div class="purchase-insight">

                        <div class="purchase-title">
                            VERISIGHT PURCHASE INSIGHT
                        </div>

                        <div class="purchase-score">
                            ${insight.score}
                            <span>/100</span>
                        </div>

                        <div class="purchase-result">
                            ${escapeHtml(insight.title)}
                        </div>

                        <div class="purchase-description">
                            ${escapeHtml(insight.description)}
                        </div>

                        <div class="purchase-metrics">

                            <div>
                                <strong>
                                    ${insight.ratingScore.toFixed(0)}%
                                </strong>
                                <span>Rating</span>
                            </div>

                            <div>
                                <strong>
                                    ${insight.reviewTrust.toFixed(0)}%
                                </strong>
                                <span>Review Trust</span>
                            </div>

                            <div>
                                <strong>
                                    ${insight.humanScore.toFixed(0)}%
                                </strong>
                                <span>Human Reviews</span>
                            </div>

                        </div>

                        <div class="purchase-note">
                            VeriSight score is based on the analyzed
                            product and review signals.
                        </div>

                    </div>

                </div>
            `;
        }


        // ---------------------------------------------------------
        // CUSTOMERS SAY
        // ---------------------------------------------------------

        const topics = Object.keys(customersSay);

        if (topics.length > 0) {

            html += `
                <div class="section">

                    <div class="section-title">
                        💬 CUSTOMERS SAY
                    </div>

                    <div class="customers-grid">
            `;

            topics.forEach(topic => {

                const item = customersSay[topic] || {};

                if (!item.found) {
                    return;
                }

                html += `
                    <div class="customer-card">

                        <div class="customer-topic">
                            ${escapeHtml(topic)}
                        </div>

                        <div class="customer-summary">
                            ${escapeHtml(
                                item.summary || "No summary available"
                            )}
                        </div>

                        ${
                            item.positive !== null &&
                            item.positive !== undefined
                            ? `
                                <span class="positive-chip">
                                    👍 ${item.positive} positive
                                </span>
                            `
                            : ""
                        }

                        ${
                            item.negative !== null &&
                            item.negative !== undefined
                            ? `
                                <span class="negative-chip">
                                    👎 ${item.negative} negative
                                </span>
                            `
                            : ""
                        }

                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }


        // ---------------------------------------------------------
        // REVIEW CARDS
        // ---------------------------------------------------------

        if (reviews.length > 0) {

            html += `
                <div class="section">

                    <div class="section-title">
                        🔎 REVIEW ANALYSIS
                    </div>
            `;

            reviews.forEach((review, index) => {

                const aiProbability =
                    Number(review.ai_probability || 0);

                const spamProbability =
                    Number(review.spam_probability || 0);


                const riskLevel =
                    calculateRisk(
                        aiProbability,
                        spamProbability
                    );


                html += `
                    <div class="review-card">

                        <div class="review-header">

                            <div class="review-number">
                                REVIEW ${index + 1}
                            </div>

                            <div class="risk-badge ${riskLevel.class}">
                                ${riskLevel.label}
                            </div>

                        </div>


                        <div class="review-meta">

                            ${
                                review.rating !== undefined
                                ? `
                                <span>
                                    ⭐ ${escapeHtml(
                                        String(review.rating)
                                    )}
                                </span>
                                `
                                : ""
                            }

                            ${
                                review.date
                                ? `
                                <span>
                                    📅 ${escapeHtml(
                                        String(review.date)
                                    )}
                                </span>
                                `
                                : ""
                            }

                            ${
                                review.verified_purchase
                                ? `
                                <span class="verified">
                                    ✓ Verified
                                </span>
                                `
                                : ""
                            }

                        </div>


                        <div class="review-text">
                            ${escapeHtml(
                                review.review_text || ""
                            )}
                        </div>


                        <div class="classification-row">

                            <div class="classification">
                                <span>AUTHOR</span>
                                <strong>
                                    ${escapeHtml(
                                        review.author_class ||
                                        "UNKNOWN"
                                    )}
                                </strong>
                            </div>

                            <div class="classification">
                                <span>INTEGRITY</span>
                                <strong>
                                    ${escapeHtml(
                                        review.integrity_class ||
                                        "UNKNOWN"
                                    )}
                                </strong>
                            </div>

                        </div>


                        <div class="probability-section">

                            <div class="probability-row">

                                <span>
                                    🤖 AI Probability
                                </span>

                                <strong>
                                    ${aiProbability.toFixed(2)}%
                                </strong>

                            </div>

                            <div class="probability-bar">

                                <div
                                    class="probability-fill ai-fill"
                                    style="width:${Math.min(
                                        aiProbability,
                                        100
                                    )}%"
                                ></div>

                            </div>


                            <div class="probability-row">

                                <span>
                                    🚨 Spam Review Probability
                                </span>

                                <strong>
                                    ${spamProbability.toFixed(2)}%
                                </strong>

                            </div>

                            <div class="probability-bar">

                                <div
                                    class="probability-fill spam-fill"
                                    style="width:${Math.min(
                                        spamProbability,
                                        100
                                    )}%"
                                ></div>

                            </div>

                        </div>

                    </div>
                `;
            });

            html += `
                </div>
            `;
        }


        results.innerHTML = html;
    }


    // -------------------------------------------------------------
    // RISK
    // -------------------------------------------------------------

    function calculateRisk(aiProbability, spamProbability) {

        if (
            spamProbability >= 40 ||
            aiProbability >= 60
        ) {
            return {
                label: "HIGH RISK",
                class: "risk-high"
            };
        }

        if (
            spamProbability >= 20 ||
            aiProbability >= 30
        ) {
            return {
                label: "MEDIUM",
                class: "risk-medium"
            };
        }

        return {
            label: "LOW",
            class: "risk-low"
        };
    }


    // -------------------------------------------------------------
    // PURCHASE INSIGHT
    // -------------------------------------------------------------

    function calculatePurchaseInsight(product, summary) {

        const rating =
            Number(product.overall_rating || 0);

        const total =
            Number(summary.total_reviews || 0);

        const genuine =
            Number(summary.genuine_reviews || 0);

        const human =
            Number(summary.human_reviews || 0);


        const ratingScore =
            Math.min(
                Math.max(
                    (rating / 5) * 100,
                    0
                ),
                100
            );


        const reviewTrust =
            total > 0
            ? (genuine / total) * 100
            : 0;


        const humanScore =
            total > 0
            ? (human / total) * 100
            : 0;


        const finalScore =
            (
                ratingScore * 0.50
            ) +
            (
                reviewTrust * 0.30
            ) +
            (
                humanScore * 0.20
            );


        let title;
        let description;


        if (finalScore >= 80) {

            title = "GOOD TO CONSIDER";

            description =
                "The product rating and analyzed review signals are relatively positive.";

        } else if (finalScore >= 65) {

            title = "REVIEW CAREFULLY";

            description =
                "The product has useful positive signals, but some review integrity signals deserve attention.";

        } else {

            title = "HIGH CAUTION";

            description =
                "The analyzed review signals contain notable integrity concerns.";

        }


        return {
            score: finalScore.toFixed(0),
            ratingScore,
            reviewTrust,
            humanScore,
            title,
            description
        };
    }


    // -------------------------------------------------------------
    // HTML ESCAPE
    // -------------------------------------------------------------

    function escapeHtml(value) {

        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

});