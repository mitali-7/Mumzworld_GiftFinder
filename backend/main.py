from dotenv import load_dotenv
load_dotenv()
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(title="Mumzworld Gift Finder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from openai import OpenAI
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
# Load product catalog once at startup
DATA_PATH = Path(__file__).parent.parent / "data" / "products.json"
with open(DATA_PATH) as f:
    PRODUCTS = json.load(f)

# ──────────────────────────────────────────────
# Pydantic schemas (structured output contract)
# ──────────────────────────────────────────────

class GiftQuery(BaseModel):
    query: str  # e.g. "gift for a 6-month-old under 200 AED"


class ProductRecommendation(BaseModel):
    id: str
    name: str
    name_ar: str
    price_aed: float
    reasoning: str          # Why this product suits the query (EN)
    reasoning_ar: str       # Same reasoning in native Arabic
    moms_say: str           # Synthesized gifting-intent review blurb (EN)
    moms_say_ar: str        # Same blurb in native Arabic


class GiftFinderResponse(BaseModel):
    query_understood: str          # LLM's interpretation of what was asked (EN)
    query_understood_ar: str       # Same in Arabic
    recommendations: list[ProductRecommendation]
    out_of_scope: bool             # True if query is completely off-topic
    uncertainty_note: Optional[str] = None  # Expressed when confidence is low


# ──────────────────────────────────────────────
# Step 1 — Intent extraction
# ──────────────────────────────────────────────

import re

def extract_budget_regex(query: str):
    """
    Extract budget in AED using regex.
    Examples:
    'under 200 AED', 'below 150', 'budget 300'
    """
    patterns = [
        r'under\s*(\d+)',
        r'below\s*(\d+)',
        r'budget\s*(\d+)',
        r'(\d+)\s*aed'
    ]

    query_lower = query.lower()

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))

    return None

def extract_intent(query: str) -> dict:
    """
    Ask the LLM to parse the free-text query into structured intent.
    Returns a dict with: age_months_min, age_months_max, budget_aed, keywords, is_valid_query
    """
    system = """You are a query parser for a baby/toddler gift finder on Mumzworld, a Middle East e-commerce platform.
Extract structured intent from the user's gift query.

Return ONLY valid JSON with this exact schema:
{
  "age_months_min": <integer or null>,
  "age_months_max": <integer or null>,
  "budget_aed": <integer or null>,
  "keywords": ["keyword1", "keyword2"],
  "occasion": "<baby shower | birthday | eid | general | null>",
  "is_valid_query": <true if this is a baby/child/mom gift query, false if completely off-topic>,
  "uncertainty": "<null or a string explaining what is ambiguous>"
}

Rules:
- If age is given as months, use directly. If given as years, multiply by 12.
- If no budget is mentioned, set budget_aed to null.
- If the query has nothing to do with babies, children, or mothers, set is_valid_query to false.
- Extract meaningful product keywords (e.g. "teether", "educational", "sleep").
- Do NOT add any text outside the JSON object."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content
    # Strip markdown fences if model added them
    raw = re.sub(r"```json|```", "", raw).strip()
    intent = json.loads(raw)

    # Override budget using regex (more reliable)
    regex_budget = extract_budget_regex(query)
    if regex_budget is not None:
        intent["budget_aed"] = regex_budget

    return intent


# ──────────────────────────────────────────────
# Step 2 — Product retrieval (filter + rank)
# ──────────────────────────────────────────────

def retrieve_products(intent: dict, top_k: int = 5) -> list[dict]:
    """
    Filter by age + budget, then rank by TF-IDF cosine similarity to query keywords.
    Returns top_k products with their gifting-intent reviews attached.
    """
    age_min = intent.get("age_months_min")
    age_max = intent.get("age_months_max")
    budget = intent.get("budget_aed")
    keywords = intent.get("keywords", [])

    # Hard filter: age and budget
    candidates = []
    for p in PRODUCTS:
        # Age filter: product range must overlap with query range
        if age_min is not None and p["age_max_months"] < age_min:
            continue
        if age_max is not None and p["age_min_months"] > age_max:
            continue
        # Budget filter
        if budget is not None and p["price_aed"] > budget:
            continue
        candidates.append(p)

    if not candidates:
        return []

    # Soft ranking: TF-IDF over tags + description + category
    def product_text(p):
        return " ".join(p["tags"]) + " " + p["description"] + " " + p["category"]

    query_text = " ".join(keywords) if keywords else "gift"
    corpus = [product_text(p) for p in candidates]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus + [query_text])
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked = [candidates[i] for i in ranked_indices]
    except Exception:
        # Fallback if TF-IDF fails (e.g. empty corpus)
        ranked = candidates[:top_k]

    # Attach only gifting-intent reviews
    for p in ranked:
        p["gift_reviews"] = [r["text"] for r in p["reviews"] if r.get("gifting_intent")]

    return ranked


# ──────────────────────────────────────────────
# Step 3 — LLM recommendation generation
# ──────────────────────────────────────────────

def generate_recommendations(
    query: str,
    intent: dict,
    products: list[dict]
) -> GiftFinderResponse:
    """
    Send retrieved products + gifting reviews to LLM.
    LLM returns structured EN+AR recommendations with mom-to-mom blurbs.
    """
    product_context = json.dumps([
        {
            "id": p["id"],
            "name": p["name"],
            "name_ar": p["name_ar"],
            "price_aed": p["price_aed"],
            "description": p["description"],
            "gift_reviews": p.get("gift_reviews", [])
        }
        for p in products
    ], ensure_ascii=False, indent=2)

    system = """You are a bilingual gift recommendation assistant for Mumzworld, a Middle East baby e-commerce platform.
You recommend baby and toddler gifts based on a natural-language query.
You write in fluent, warm, native-sounding English AND native Gulf Arabic (not a literal translation — natural register for a Gulf Arab mom). Write Arabic in natural, conversational Gulf dialect used by mothers. Avoid literal translations.

CRITICAL RULES:
1. ONLY recommend products from the provided catalog. Never invent products.
2. If no products fit, return an empty recommendations list and set out_of_scope or uncertainty_note.
3. The moms_say field MUST be derived from the gift_reviews provided. Do not invent reviews.
   If no gift reviews exist for a product, write: "No gift reviews yet for this product."
4. Arabic text must read naturally — avoid word-for-word translations. Use vocabulary a Gulf mom would actually use.
5. Express uncertainty honestly. Do not pad output with generic claims.

FINAL QUALITY RULES:

1. Arabic:
- Use simple, natural spoken Arabic
- Avoid formal phrases like "تخضع للميزانية"
- Use casual phrasing like:
  "مناسبة للميزانية"
  "الأطفال يحبونها"
  "مرة عملية"

2. Moms Say:
- DO NOT start with "Moms love this"
- Start directly with real insights from reviews
- Use specific experiences from reviews
- Avoid exaggeration or invented claims

3. Grounding:
- Only use facts present in reviews
- Do not generalize beyond input data

4. Reasoning:
- Be specific to the product
- Avoid generic phrases like "good for development"

Return ONLY valid JSON matching this schema exactly:
{
  "query_understood": "<1 sentence: what you understood the request to be, in English>",
  "query_understood_ar": "<same sentence in native Arabic>",
  "out_of_scope": <true | false>,
  "uncertainty_note": "<null or a sentence about what is unclear>",
  "recommendations": [
    {
      "id": "<product id>",
      "name": "<product name in English>",
      "name_ar": "<product name in Arabic>",
      "price_aed": <number>,
      "reasoning": "<1-2 sentences: why this fits the query, in English>",
      "reasoning_ar": "<same reasoning in native Arabic>",
      "moms_say": "<2-3 sentences synthesizing what moms say about this as a gift, in English. Grounded in reviews.>",
      "moms_say_ar": "<same synthesis in native Arabic>"
    }
  ]
}

Do NOT include any text outside the JSON object."""

    user_message = f"""Query: {query}

Parsed intent: {json.dumps(intent, ensure_ascii=False)}

Available products from catalog:
{product_context}

Generate recommendations now."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        data = json.loads(raw)
        return GiftFinderResponse(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Schema validation failed: {e}\nRaw LLM output: {raw}")


# ──────────────────────────────────────────────
# API endpoint
# ──────────────────────────────────────────────

@app.post("/find-gifts", response_model=GiftFinderResponse)
async def find_gifts(body: GiftQuery):
    query = body.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if len(query) > 500:
        raise HTTPException(status_code=400, detail="Query too long. Keep it under 500 characters.")

    # Step 1: Parse intent
    try:
        intent = extract_intent(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent extraction failed: {e}")

    # If LLM flagged query as off-topic, short-circuit
    if not intent.get("is_valid_query", True):
        return GiftFinderResponse(
            query_understood="This query does not appear to be about baby or toddler gifts.",
            query_understood_ar="يبدو أن هذا الطلب لا يتعلق بهدايا الأطفال أو الرضّع.",
            recommendations=[],
            out_of_scope=True,
            uncertainty_note="Please ask about gifts for babies, toddlers, or mothers."
        )

    # Step 2: Retrieve products
    products = retrieve_products(intent, top_k=4)

    if not products:
        return GiftFinderResponse(
            query_understood=f"Gift search for: {query}",
            query_understood_ar=f"البحث عن هدية: {query}",
            recommendations=[],
            out_of_scope=False,
            uncertainty_note="No products in our catalog match your age range and budget. Try adjusting your budget or age range."
        )

    # Step 3: Generate structured bilingual recommendations
    try:
        result = generate_recommendations(query, intent, products)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "products_loaded": len(PRODUCTS)}