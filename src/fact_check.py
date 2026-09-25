import os
import re
import requests
from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv(
    "FACT_CHECK_API_KEY"
)

FACT_CHECK_URL = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(text):
    """
    Normalize text so we can compare the user's
    claim with returned fact-check claims.
    """

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
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
# GET MEANINGFUL WORDS
# ==========================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "with", "by",
    "about", "as", "that", "this", "it", "its",
    "has", "have", "had", "does", "do", "did",
    "will", "would", "could", "should", "can",
    "may", "might", "than", "then", "only",
    "very", "just", "also", "into", "their",
    "they", "them", "he", "she", "we", "you",
    "i", "my", "your", "our", "his", "her",
    "today"
}


def get_keywords(text):
    """
    Extract meaningful words from text.
    """

    normalized = normalize_text(
        text
    )

    words = normalized.split()

    keywords = {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }

    return keywords


# ==========================================
# CLAIM DECOMPOSITION
# ==========================================

MAX_CLAIMS = 5


def clean_claim_part(text):
    """
    Clean an individual claim after splitting it.
    """

    text = text.strip()

    # Remove unnecessary punctuation from
    # the beginning and end.
    text = re.sub(
        r"^[\s,;:]+|[\s,;:]+$",
        "",
        text
    )

    # Remove repeated spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def decompose_claims(text):
    """
    Break a long statement into smaller individual
    claims.

    This is a rule-based first version.

    It primarily uses:
    - Sentence boundaries
    - Commas followed by conjunctions
    - Common conjunctions such as 'and', 'but',
      and 'while'

    Maximum number of returned claims:
    MAX_CLAIMS
    """

    if not text:
        return []

    text = text.strip()

    if not text:
        return []


    # ======================================
    # STEP 1: SPLIT INTO SENTENCES
    # ======================================

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    first_parts = []

    for sentence in sentences:

        sentence = clean_claim_part(
            sentence
        )

        if sentence:
            first_parts.append(
                sentence
            )


    # ======================================
    # STEP 2: SPLIT COMPOUND SENTENCES
    # ======================================

    claims = []


    for sentence in first_parts:

        # First look for a comma followed by
        # a conjunction.
        parts = re.split(
            r",\s+(?=(?:and|but|while|also)\b)",
            sentence,
            flags=re.IGNORECASE
        )


        for part in parts:

            part = clean_claim_part(
                part
            )

            if not part:
                continue


            # ==================================
            # STEP 3: SPLIT STRONG CONJUNCTIONS
            # ==================================

            # We only split when the conjunction
            # appears to connect two reasonably
            # large pieces of information.

            # This helps avoid unnecessarily
            # splitting very short phrases.

            conjunction_parts = re.split(
                r"\s+(?:and|but|while)\s+",
                part,
                flags=re.IGNORECASE
            )


            for sub_part in conjunction_parts:

                sub_part = clean_claim_part(
                    sub_part
                )

                if not sub_part:
                    continue


                # Remove conjunctions left at the
                # beginning of an individual claim.
                sub_part = re.sub(
                    r"^(?:and|but|while|also)\s+",
                    "",
                    sub_part,
                    flags=re.IGNORECASE
                ).strip()


                if not sub_part:
                    continue


                word_count = len(
                    sub_part.split()
                )


                # Ignore extremely short fragments.
                if word_count < 4:
                    continue


                claims.append(
                    sub_part
                )


    # ======================================
    # STEP 4: REMOVE DUPLICATES
    # ======================================

    unique_claims = []

    seen = set()


    for claim in claims:

        normalized_claim = normalize_text(
            claim
        )


        if not normalized_claim:
            continue


        if normalized_claim in seen:
            continue


        seen.add(
            normalized_claim
        )


        unique_claims.append(
            claim
        )


    # ======================================
    # STEP 5: LIMIT CLAIM COUNT
    # ======================================

    unique_claims = unique_claims[
        :MAX_CLAIMS
    ]


    # ======================================
    # FALLBACK
    # ======================================

    # If decomposition produced nothing,
    # keep the original text as one claim.

    if not unique_claims:

        return [
            text
        ]


    return unique_claims


# ==========================================
# CALCULATE RELEVANCE
# ==========================================

def calculate_relevance(query, fact_check):
    """
    Calculate how closely a returned fact-check
    relates to the user's original claim.

    This prevents unrelated fact-checks from
    being displayed as evidence.
    """

    query_keywords = get_keywords(
        query
    )

    fact_check_text = " ".join([
        fact_check.get(
            "claim",
            ""
        ),
        fact_check.get(
            "title",
            ""
        )
    ])

    fact_keywords = get_keywords(
        fact_check_text
    )

    if not query_keywords or not fact_keywords:
        return 0.0

    overlap = query_keywords.intersection(
        fact_keywords
    )

    overlap_count = len(
        overlap
    )

    # Percentage of user's meaningful words
    # that appear in the fact-check.
    query_coverage = (
        overlap_count /
        len(query_keywords)
    )

    # Percentage of fact-check keywords
    # that appear in the user's claim.
    fact_coverage = (
        overlap_count /
        len(fact_keywords)
    )

    # Balanced score
    if query_coverage + fact_coverage == 0:

        score = 0.0

    else:

        score = (
            2 *
            query_coverage *
            fact_coverage
        ) / (
            query_coverage +
            fact_coverage
        )

    return round(
        score,
        3
    )


# ==========================================
# SEARCH FACT CHECK API
# ==========================================

def search_fact_checks(query):

    if not API_KEY:

        return {
            "success": False,
            "error":
                "Fact Check API key is not configured."
        }


    params = {
        "key": API_KEY,
        "query": query,
        "languageCode": "en",
        "pageSize": 5
    }


    try:

        response = requests.get(
            FACT_CHECK_URL,
            params=params,
            timeout=10
        )


        if response.status_code != 200:

            return {
                "success": False,
                "error": (
                    "Fact Check API returned "
                    f"status {response.status_code}."
                )
            }


        data = response.json()


        claims = data.get(
            "claims",
            []
        )


        results = []


        for claim in claims:

            claim_text = claim.get(
                "text",
                ""
            )


            reviews = claim.get(
                "claimReview",
                []
            )


            for review in reviews:

                publisher = review.get(
                    "publisher",
                    {}
                )


                results.append({

                    "claim": claim_text,

                    "rating": review.get(
                        "textualRating",
                        "Unknown"
                    ),

                    "publisher": publisher.get(
                        "name",
                        "Unknown"
                    ),

                    "publisher_site": publisher.get(
                        "site",
                        ""
                    ),

                    "title": review.get(
                        "title",
                        ""
                    ),

                    "url": review.get(
                        "url",
                        ""
                    ),

                    "review_date": review.get(
                        "reviewDate",
                        ""
                    )

                })


        return {
            "success": True,
            "results": results
        }


    except requests.RequestException as error:

        return {
            "success": False,
            "error": str(error)
        }


# ==========================================
# VERIFY CLAIM
# ==========================================

def verify_claim(query):
    """
    Search for fact-checks and return only
    results that are sufficiently relevant
    to the user's actual claim.
    """

    query = query.strip()


    if not query:
        return []


    # Search the original claim and a shortened
    # version for long articles.
    queries = [
        query,
        query[:250].strip()
    ]


    all_results = []


    for search_query in queries:

        if not search_query:
            continue


        response = search_fact_checks(
            search_query
        )


        if response.get(
            "success"
        ):

            results = response.get(
                "results",
                []
            )


            all_results.extend(
                results
            )


    # ======================================
    # SCORE EACH RESULT
    # ======================================

    scored_results = []


    for result in all_results:

        relevance = calculate_relevance(
            query,
            result
        )


        result["relevance"] = relevance


        # Only accept sufficiently relevant
        # fact-checks.
        #
        # This is the important protection
        # against unrelated results.
        if relevance >= 0.20:

            scored_results.append(
                result
            )


    # ======================================
    # REMOVE DUPLICATES
    # ======================================

    unique_results = []

    seen_urls = set()


    for result in scored_results:

        url = result.get(
            "url",
            ""
        )


        if url and url not in seen_urls:

            seen_urls.add(
                url
            )


            unique_results.append(
                result
            )


    # ======================================
    # SORT BY RELEVANCE
    # ======================================

    unique_results.sort(
        key=lambda item: item.get(
            "relevance",
            0
        ),
        reverse=True
    )


    return unique_results


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )


    print(
        "CLAIM DECOMPOSITION TEST"
    )


    print(
        "=============================="
    )


    test_claim = (
        "NASA confirmed that the Earth is flat, "
        "and scientists discovered that gravity "
        "does not exist."
    )


    print(
        "\nOriginal statement:"
    )


    print(
        test_claim
    )


    claims = decompose_claims(
        test_claim
    )


    print(
        "\nIndividual claims:"
    )


    for index, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"{index}. {claim}"
        )


    print(
        "\n=============================="
    )


    print(
        "FACT CHECK TEST"
    )


    print(
        "=============================="
    )


    fact_check_test = (
        "The Earth will end in 3 days."
    )


    result = verify_claim(
        fact_check_test
    )


    print(
        "\nFact-check results:"
    )


    for item in result:

        print(
            "\nTitle:",
            item.get(
                "title"
            )
        )


        print(
            "Claim:",
            item.get(
                "claim"
            )
        )


        print(
            "Publisher:",
            item.get(
                "publisher"
            )
        )


        print(
            "Relevance:",
            item.get(
                "relevance"
            )
        )


        print(
            "Rating:",
            item.get(
                "rating"
            )
        )