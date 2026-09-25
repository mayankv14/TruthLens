from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import joblib
import re

from src.database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    verify_password,
    save_prediction,
    get_user_predictions
)

from src.auth import (
    create_token,
    login_required
)

from src.fact_check import (
    verify_claim,
    decompose_claims
)


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# LOAD ML MODEL
# ==========================================

vectorizer = joblib.load(
    "model/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "model/fake_news_model.pkl"
)


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# ANALYZE SINGLE CLAIM
# ==========================================

def analyze_single_claim(claim):
    """
    Analyze one individual claim.

    The verification order is:

    1. Fact-check evidence
    2. If no evidence and claim is short -> UNCERTAIN
    3. Otherwise -> ML analysis
    """

    claim = claim.strip()

    if not claim:

        return {
            "claim": claim,
            "status": "UNCERTAIN",
            "confidence": None,
            "fact_checks": [],
            "reason": "Empty claim."
        }


    # ======================================
    # FACT CHECK FIRST
    # ======================================

    fact_checks = verify_claim(
        claim
    )


    if fact_checks:

        return {
            "claim": claim,

            "status":
            "FACT_CHECK_FOUND",

            "confidence":
            None,

            "fact_checks":
            fact_checks,

            "reason": (
                "A relevant fact-check was found "
                "for this individual claim."
            )
        }


    # ======================================
    # CLEAN CLAIM
    # ======================================

    cleaned_claim = clean_text(
        claim
    )


    words = cleaned_claim.split()


    # ======================================
    # SHORT / UNKNOWN CLAIM
    # ======================================

    if len(words) < 8:

        return {
            "claim": claim,

            "status":
            "UNCERTAIN",

            "confidence":
            None,

            "fact_checks":
            [],

            "reason": (
                "No matching fact-check was found "
                "and the claim is too short for a "
                "reliable machine-learning assessment."
            )
        }


    # ======================================
    # ML ANALYSIS
    # ======================================

    claim_tfidf = vectorizer.transform(
        [cleaned_claim]
    )


    prediction = model.predict(
        claim_tfidf
    )[0]


    probability = model.predict_proba(
        claim_tfidf
    )[0]


    confidence = round(
        max(probability) * 100,
        2
    )


    if prediction == 1:

        status = "LIKELY_REAL"

    else:

        status = "LIKELY_FAKE"


    return {

        "claim":
        claim,

        "status":
        status,

        "confidence":
        confidence,

        "fact_checks":
        [],

        "reason": (
            "No matching fact-check was found. "
            "This individual claim was analyzed "
            "using the trained machine-learning model."
        )
    }


# ==========================================
# BUILD CLAIM DECOMPOSITION
# ==========================================

def build_claim_decomposition(original_news):
    """
    Break the user's input into individual claims
    and analyze each claim separately.
    """

    claims = decompose_claims(
        original_news
    )


    decomposition = []


    for index, claim in enumerate(
        claims,
        start=1
    ):

        analysis = analyze_single_claim(
            claim
        )


        decomposition.append({

            "claim_number":
            index,

            "claim":
            analysis["claim"],

            "status":
            analysis["status"],

            "confidence":
            analysis["confidence"],

            "reason":
            analysis["reason"],

            "fact_checks":
            analysis["fact_checks"]

        })


    return decomposition


# ==========================================
# BUILD OVERALL RESULT FROM CLAIMS
# ==========================================

def build_overall_result(
    claim_decomposition
):
    """
    Determine the overall result from the
    individual claim assessments.

    For multiple claims:

    - External fact-check evidence has priority.
    - Mixed evidence results in UNCERTAIN.
    - Different ML classifications result in
      UNCERTAIN.
    - Confidence is not averaged across claims
      because that would create a misleading
      overall probability.
    """

    if not claim_decomposition:

        return {

            "prediction":
            "UNCERTAIN",

            "confidence":
            None,

            "reason": (
                "No individual claims were available "
                "for analysis."
            )

        }


    statuses = [

        item.get(
            "status",
            "UNCERTAIN"
        )

        for item in claim_decomposition

    ]


    # ======================================
    # SINGLE CLAIM
    # ======================================

    if len(statuses) == 1:

        status = statuses[0]


        return {

            "prediction":
            status,

            "confidence":
            claim_decomposition[0].get(
                "confidence"
            ),

            "reason":
            claim_decomposition[0].get(
                "reason",
                "No additional assessment available."
            )

        }


    # ======================================
    # MULTIPLE CLAIMS
    # ======================================

    fact_check_count = statuses.count(
        "FACT_CHECK_FOUND"
    )


    uncertain_count = statuses.count(
        "UNCERTAIN"
    )


    likely_real_count = statuses.count(
        "LIKELY_REAL"
    )


    likely_fake_count = statuses.count(
        "LIKELY_FAKE"
    )


    total_claims = len(
        statuses
    )


    # ======================================
    # ALL CLAIMS HAVE FACT-CHECK EVIDENCE
    # ======================================

    if fact_check_count == total_claims:

        return {

            "prediction":
            "FACT_CHECK_FOUND",

            "confidence":
            None,

            "reason": (
                f"All {total_claims} individual claims "
                "have matching external fact-check evidence. "
                "Review the cited sources for each claim."
            )

        }


    # ======================================
    # MIXED FACT-CHECK + OTHER RESULTS
    # ======================================

    if fact_check_count > 0:

        return {

            "prediction":
            "UNCERTAIN",

            "confidence":
            None,

            "reason": (
                f"{fact_check_count} of {total_claims} "
                "individual claims have matching external "
                "fact-check evidence, while the remaining "
                "claims do not have equivalent external "
                "evidence. The overall statement is therefore "
                "not assigned a single ML label."
            )

        }


    # ======================================
    # ANY CLAIM IS UNCERTAIN
    # ======================================

    if uncertain_count > 0:

        return {

            "prediction":
            "UNCERTAIN",

            "confidence":
            None,

            "reason": (
                f"The statement contains {total_claims} "
                "individual claims, and at least one could "
                "not be reliably assessed. TruthLens therefore "
                "does not assign a single overall classification."
            )

        }


    # ======================================
    # ALL ML CLAIMS ARE LIKELY REAL
    # ======================================

    if likely_real_count == total_claims:

        return {

            "prediction":
            "LIKELY_REAL",

            "confidence":
            None,

            "reason": (
                f"All {total_claims} individual claims were "
                "classified as LIKELY_REAL by the machine-learning "
                "model. This is a model assessment, not factual proof."
            )

        }


    # ======================================
    # ALL ML CLAIMS ARE LIKELY FAKE
    # ======================================

    if likely_fake_count == total_claims:

        return {

            "prediction":
            "LIKELY_FAKE",

            "confidence":
            None,

            "reason": (
                f"All {total_claims} individual claims were "
                "classified as LIKELY_FAKE by the machine-learning "
                "model. This is a model assessment, not factual proof."
            )

        }


    # ======================================
    # MIXED ML RESULTS
    # ======================================

    return {

        "prediction":
        "UNCERTAIN",

        "confidence":
        None,

        "reason": (
            f"The statement contains {total_claims} claims "
            "with mixed machine-learning assessments. TruthLens "
            "does not combine different claim classifications "
            "into a single definitive result."
        )

    }


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# REGISTER PAGE
# ==========================================

@app.route("/register")
def register_page():

    return render_template(
        "register.html"
    )


# ==========================================
# LOGIN PAGE
# ==========================================

@app.route("/login")
def login_page():

    return render_template(
        "login.html"
    )


# ==========================================
# AUTHENTICATION
# ==========================================


# ==========================================
# REGISTER
# ==========================================

@app.route(
    "/api/auth/register",
    methods=["POST"]
)
def register():

    data = request.get_json(
        silent=True
    ) or {}


    name = str(
        data.get("name", "")
    ).strip()


    email = str(
        data.get("email", "")
    ).strip().lower()


    password = str(
        data.get("password", "")
    )


    # ======================================
    # VALIDATION
    # ======================================

    if not name:

        return jsonify({

            "error":
            "Name is required."

        }), 400


    if not email:

        return jsonify({

            "error":
            "Email is required."

        }), 400


    if not password:

        return jsonify({

            "error":
            "Password is required."

        }), 400


    if len(password) < 6:

        return jsonify({

            "error":
            "Password must be at least 6 characters."

        }), 400


    # ======================================
    # CHECK EXISTING USER
    # ======================================

    existing_user = get_user_by_email(
        email
    )


    if existing_user:

        return jsonify({

            "error":
            "An account with this email already exists."

        }), 409


    # ======================================
    # CREATE USER
    # ======================================

    user_id = create_user(
        name,
        email,
        password
    )


    if not user_id:

        return jsonify({

            "error":
            "Unable to create account."

        }), 500


    # ======================================
    # CREATE JWT
    # ======================================

    token = create_token(
        user_id
    )


    return jsonify({

        "message":
        "Account created successfully.",

        "token":
        token,

        "user": {

            "id":
            user_id,

            "name":
            name,

            "email":
            email

        }

    }), 201


# ==========================================
# LOGIN
# ==========================================

@app.route(
    "/api/auth/login",
    methods=["POST"]
)
def login():

    data = request.get_json(
        silent=True
    ) or {}


    email = str(
        data.get("email", "")
    ).strip().lower()


    password = str(
        data.get("password", "")
    )


    # ======================================
    # VALIDATION
    # ======================================

    if not email or not password:

        return jsonify({

            "error":
            "Email and password are required."

        }), 400


    # ======================================
    # FIND USER
    # ======================================

    user = get_user_by_email(
        email
    )


    if not user:

        return jsonify({

            "error":
            "Invalid email or password."

        }), 401


    # ======================================
    # CHECK PASSWORD
    # ======================================

    password_valid = verify_password(
        password,
        user["password_hash"]
    )


    if not password_valid:

        return jsonify({

            "error":
            "Invalid email or password."

        }), 401


    # ======================================
    # CREATE JWT
    # ======================================

    user_id = str(
        user["_id"]
    )


    token = create_token(
        user_id
    )


    return jsonify({

        "message":
        "Login successful.",

        "token":
        token,

        "user": {

            "id":
            user_id,

            "name":
            user["name"],

            "email":
            user["email"]

        }

    })


# ==========================================
# CURRENT USER
# ==========================================

@app.route(
    "/api/auth/me",
    methods=["GET"]
)
@login_required
def current_user():

    user = get_user_by_id(
        request.user_id
    )


    if not user:

        return jsonify({

            "error":
            "User not found."

        }), 404


    return jsonify({

        "user": {

            "id":
            str(user["_id"]),

            "name":
            user["name"],

            "email":
            user["email"]

        }

    })


# ==========================================
# PREDICT / VERIFY
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    data = request.get_json(
        silent=True
    ) or {}


    news = str(
        data.get("news", "")
    )


    # ======================================
    # EMPTY INPUT
    # ======================================

    if not news.strip():

        return jsonify({

            "error":
            "Please enter a news claim or article."

        }), 400


    original_news = news.strip()


    # ======================================
    # CLAIM DECOMPOSITION
    # ======================================

    claim_decomposition = (
        build_claim_decomposition(
            original_news
        )
    )


    # ======================================
    # BUILD OVERALL RESULT
    # FROM INDIVIDUAL CLAIMS
    # ======================================

    overall = build_overall_result(
        claim_decomposition
    )


    result = overall[
        "prediction"
    ]


    confidence = overall[
        "confidence"
    ]


    reason = overall[
        "reason"
    ]


    # ======================================
    # COLLECT FACT-CHECK EVIDENCE
    # FROM INDIVIDUAL CLAIMS
    # ======================================

    fact_checks = []

    seen_fact_check_urls = set()


    for item in claim_decomposition:

        for fact_check in item.get(
            "fact_checks",
            []
        ):

            url = fact_check.get(
                "url",
                ""
            )


            # Avoid displaying the same
            # source multiple times.
            if url:

                if url in seen_fact_check_urls:
                    continue

                seen_fact_check_urls.add(
                    url
                )


            fact_checks.append(
                fact_check
            )


    # ======================================
    # SAVE RESULT
    # ======================================

    save_prediction(
        original_news,
        result,
        confidence,
        request.user_id
    )


    # ======================================
    # INVESTIGATION RESPONSE
    # ======================================

    if len(claim_decomposition) > 1:

        fact_check_count = sum(

            1

            for item in claim_decomposition

            if item.get("status") ==
            "FACT_CHECK_FOUND"

        )


        uncertain_count = sum(

            1

            for item in claim_decomposition

            if item.get("status") ==
            "UNCERTAIN"

        )


        ml_count = sum(

            1

            for item in claim_decomposition

            if item.get("status") in [

                "LIKELY_REAL",

                "LIKELY_FAKE"

            ]

        )


        if fact_check_count > 0:

            verification_method = (
                "Claim-level external "
                "fact-check evidence"
            )


            evidence_status = (

                f"{fact_check_count} of "
                f"{len(claim_decomposition)} "
                "claims have matching "
                "fact-check evidence"

            )


        elif ml_count > 0:

            verification_method = (
                "Claim-level machine-learning analysis"
            )


            evidence_status = (
                "No matching external fact-check "
                "evidence found"
            )


        else:

            verification_method = (
                "Insufficient evidence"
            )


            evidence_status = (
                "No reliable evidence found"
            )


        if result == "FACT_CHECK_FOUND":

            summary = (
                "TruthLens analyzed each individual "
                "claim and found matching external "
                "fact-check evidence for all claims."
            )


        elif result == "UNCERTAIN":

            summary = (
                "TruthLens separated the statement "
                "into individual claims. The claims "
                "did not all have the same level of "
                "available evidence, so TruthLens did "
                "not assign one definitive overall "
                "classification."
            )


        else:

            summary = (
                "TruthLens analyzed the individual "
                "claims separately and reached the "
                "same machine-learning assessment "
                "for each claim."
            )


        investigation = {

            "claim":
            original_news,

            "verification_method":
            verification_method,

            "evidence_status":
            evidence_status,

            "model_used":
            ml_count > 0,

            "model_assessment":
            result,

            "model_confidence":
            confidence,

            "summary":
            summary,

            "explanation":
            reason

        }


    else:

        single_claim = (

            claim_decomposition[0]

            if claim_decomposition

            else {}

        )


        status = single_claim.get(
            "status",
            "UNCERTAIN"
        )


        if status == "FACT_CHECK_FOUND":

            investigation = {

                "claim":
                original_news,

                "verification_method":
                "External fact-check evidence",

                "evidence_status":
                "Evidence found",

                "model_used":
                False,

                "model_assessment":
                "Not required",

                "summary": (
                    "TruthLens found relevant "
                    "published fact-check "
                    "evidence addressing this claim."
                ),

                "explanation": (
                    "The result is based on external "
                    "fact-checking evidence. Review "
                    "the cited publisher, claim "
                    "assessment, and original source "
                    "before drawing your own conclusion."
                )

            }


        elif status == "LIKELY_REAL":

            investigation = {

                "claim":
                original_news,

                "verification_method":
                "Machine-learning analysis",

                "evidence_status":
                "No matching fact-check found",

                "model_used":
                True,

                "model_assessment":
                status,

                "model_confidence":
                confidence,

                "summary": (
                    "No matching external fact-check "
                    "was found. TruthLens therefore "
                    "used its trained machine-learning "
                    "model to analyze the text."
                ),

                "explanation": (
                    "The machine-learning model identified "
                    "patterns associated with this class "
                    "in its training data. This confidence "
                    "value represents the model's "
                    "classification probability, not "
                    "factual certainty."
                )

            }


        elif status == "LIKELY_FAKE":

            investigation = {

                "claim":
                original_news,

                "verification_method":
                "Machine-learning analysis",

                "evidence_status":
                "No matching fact-check found",

                "model_used":
                True,

                "model_assessment":
                status,

                "model_confidence":
                confidence,

                "summary": (
                    "No matching external fact-check "
                    "was found. TruthLens therefore "
                    "used its trained machine-learning "
                    "model to analyze the text."
                ),

                "explanation": (
                    "The machine-learning model identified "
                    "patterns associated with this class "
                    "in its training data. This confidence "
                    "value represents the model's "
                    "classification probability, not "
                    "factual certainty."
                )

            }


        else:

            investigation = {

                "claim":
                original_news,

                "verification_method":
                "Insufficient evidence",

                "evidence_status":
                "No matching evidence found",

                "model_used":
                False,

                "model_assessment":
                "Not performed",

                "summary": (
                    "TruthLens did not find matching "
                    "external fact-check evidence and "
                    "did not have enough information "
                    "for a reliable machine-learning "
                    "assessment."
                ),

                "explanation": (
                    "TruthLens does not have enough "
                    "available evidence to make a "
                    "dependable classification for "
                    "this claim."
                )

            }


    # ======================================
    # FINAL RESPONSE
    # ======================================

    return jsonify({

        "prediction":
        result,

        "confidence":
        confidence,

        "reason":
        reason,

        "fact_checks":
        fact_checks,

        "investigation":
        investigation,

        "claim_decomposition":
        claim_decomposition

    })


# ==========================================
# HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    return render_template(
        "history.html"
    )


# ==========================================
# HISTORY API
# ==========================================

@app.route(
    "/api/history",
    methods=["GET"]
)
@login_required
def api_history():

    records = get_user_predictions(
        request.user_id
    )


    return jsonify(
        records
    )


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )