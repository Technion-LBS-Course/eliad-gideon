"""M4 agent layer — translates a user's free-text profile into the four parameters
the M3 model needs, runs YOUR model, and returns ranked venues.

The loop (the number ALWAYS comes from the model, never the LLM):
  1. user describes themselves in free text / a profile form
  2. the LLM classifies that into 4 parameters (city, budget, quality, user_type) as JSON
  3. the saved KMeans model predicts which cluster the user's ideal profile falls in, and we
     return real venues the model placed in that same cluster (the model — not the LLM and not
     a hand-coded formula — selects the group)
  4. we present the top-5 venues; on any invalid LLM output we fall back (ask for a city,
     return best / cheapest / closest)
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.model import FEATURE_COLS, PERSONA_WEIGHTS, _haversine, assign_cluster_labels

# Groq is OpenAI-compatible (NOT Anthropic) — use chat.completions.create.
MODEL_NAME = "llama-3.3-70b-versatile"

VALID_USER_TYPES = ("student", "quality")
VALID_QUALITY = ("low", "medium", "high")

# Quality preference (איכות) → minimum raw rating the venue must clear (hard floor).
QUALITY_MIN_RATING = {"low": 0.0, "medium": 4.0, "high": 4.5}

# Quality preference → which quantile of the *actual* rating distribution to use as
# the rating coordinate of the user's "ideal venue". Using a real quantile (instead of a fixed
# 5.0) keeps the ideal point inside the populated region of feature space, so the model lands
# on a cluster that actually contains venues.
QUALITY_RATING_QUANTILE = {"low": 0.25, "medium": 0.55, "high": 0.90}

# Budget inferred from life status (סכום) when the user gives no explicit ceiling.
BUDGET_BY_STATUS = {"student": 58, "working": 68, "retired": 62, "other": 78}


def build_system_prompt(cities: list[str]) -> str:
    """The system prompt — written ONCE. Defines who the agent is, what is allowed,
    and the strict JSON output format (the 4 categories). The user cannot override it."""
    city_list = ", ".join(cities)
    return (
        "You are the parameter-extraction layer of 'Appetite Engineering', a shawarma "
        "recommendation app for Israel. Your ONLY job is to read a user's profile and "
        "classify it into exactly FOUR parameters that a separate ML model needs. "
        "You do NOT recommend venues, you do NOT invent names, and you do NOT compute "
        "scores — the model does all of that.\n\n"
        "Return ONLY a JSON object (no markdown fences, no prose, no trailing text) with "
        "exactly these four keys:\n"
        '  "city": string — the Israeli city the user is in (מיקום). Choose the closest '
        f"match from this list: [{city_list}]. Map a neighbourhood or landmark to its "
        "city. If it cannot be determined, use null.\n"
        '  "max_budget_nis": integer — the most the user will pay for one shawarma (סכום). '
        "If not stated, infer from life status: student≈58, working≈68, retired≈62, else 78.\n"
        '  "quality_preference": one of "low", "medium", "high" — how much the user '
        "prioritises rating/quality (איכות).\n"
        '  "user_type": one of "student", "quality" — the persona (סוג משתמש). Map '
        "young / student / tight-budget profiles to \"student\"; older / professional / "
        'quality-seeking profiles to "quality".\n\n'
        "Rules:\n"
        "- Output strictly valid JSON with double-quoted keys and string values.\n"
        "- Include all four keys, never more.\n"
        "- Base every field on the user's data; do not leave fields blank."
    )


def build_user_prompt(user_text: str) -> str:
    """The user prompt — changes every call. The user's free-text description of themselves."""
    return (
        "Here is the user describing themselves and what they want, in their own words "
        "(Hebrew or English):\n"
        f'"""\n{user_text.strip()}\n"""\n\n'
        "Classify this into the four parameters as instructed. Return JSON only."
    )


def _parse_json(text: str) -> dict | None:
    """Pull a JSON object out of the LLM reply, tolerating markdown fences / stray text."""
    if not text:
        return None
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    raw = fenced.group(0) if fenced else text
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def validate_params(raw: dict | None, valid_cities: list[str]) -> dict | None:
    """Output validation. Returns clean params, or None to trigger the fallback path."""
    if not isinstance(raw, dict):
        return None

    user_type = str(raw.get("user_type", "")).strip().lower()
    if user_type not in VALID_USER_TYPES:
        return None

    quality = str(raw.get("quality_preference", "")).strip().lower()
    if quality not in VALID_QUALITY:
        return None

    try:
        budget = int(round(float(raw.get("max_budget_nis"))))
    except (TypeError, ValueError):
        return None
    if not 20 <= budget <= 200:
        return None

    city = raw.get("city")
    city = str(city).strip() if city not in (None, "", "null") else None
    # Keep the city only if it actually exists in the dataset; otherwise drop to fallback.
    if city is not None and city not in valid_cities:
        match = next((c for c in valid_cities if c == city or city in c or c in city), None)
        city = match  # may be None → handled by caller

    return {
        "city": city,
        "max_budget_nis": budget,
        "quality_preference": quality,
        "user_type": user_type,
    }


def extract_params(user_text: str, api_key: str, valid_cities: list[str]) -> dict | None:
    """Steps 1-2: call the LLM to classify the free-text profile into the 4 params, then
    validate. Returns validated params, or None on any failure (network, bad JSON, range)."""
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,  # low temperature → reliable JSON
            messages=[
                {"role": "system", "content": build_system_prompt(valid_cities)},
                {"role": "user", "content": build_user_prompt(user_text)},
            ],
        )
        raw = _parse_json(resp.choices[0].message.content)
    except Exception:
        return None
    return validate_params(raw, valid_cities)


def _with_distance(df: pd.DataFrame, user_lat, user_lng) -> pd.DataFrame:
    df = df.copy()
    if user_lat is not None and user_lng is not None:
        df["distance_km"] = df.apply(
            lambda r: _haversine(user_lat, user_lng, r["lat"], r["lng"]), axis=1
        )
    else:
        df["distance_km"] = 0.0
    return df


def recommend(
    model_result: dict,
    df: pd.DataFrame,
    params: dict,
    user_lat: float | None = None,
    user_lng: float | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Step 3: YOUR KMeans model selects the venues.

    The user's hard constraints (city, budget) filter the candidate pool. Then we build the
    user's *ideal venue* as a point in the model's feature space and call the model's
    predict() on it — the cluster the model assigns to that ideal is the target. We return
    real venues the model placed in the same cluster. The persona weights only order venues
    *within* the model-chosen cluster; they never override which cluster the model picked.
    The returned frame carries .attrs['model_target_label'] / ['model_target_cluster']."""
    df_full = df.dropna(subset=["price_nis", "rating", "weighted_rating"]).copy()

    df = df_full
    if params.get("city"):
        df = df[df["city"] == params["city"]]
    df = df[df["price_nis"] <= params["max_budget_nis"]]
    # Hard quality floor on raw rating (weighted_rating collapses to the global mean for the
    # ~95% of venues lacking a review count, so it can't discriminate quality; rating can).
    min_rating = QUALITY_MIN_RATING.get(params["quality_preference"], 0.0)
    df = df[df["rating"] >= min_rating].copy()
    if df.empty:
        return df

    df = _with_distance(df, user_lat, user_lng)

    # The model assigns every candidate venue to a cluster (native predict()).
    scaler, model = model_result["scaler"], model_result["model"]
    df["cluster"] = model.predict(scaler.transform(df[FEATURE_COLS].fillna(0)))

    # Build the user's ideal venue in FEATURE_COLS order [price_nis, rating] and ask
    # the MODEL which cluster it belongs to. Both coordinates are real quantiles of the
    # candidate pool, so the ideal always lands in populated feature space (never an empty
    # cluster). Students anchor to the cheap end; others to the median price (price and quality
    # are uncorrelated here, r≈-0.06, so chasing a higher price buys nothing). This step is what
    # makes the model — not a formula — decide the group.
    price_q = 0.15 if params["user_type"] == "student" else 0.50
    ideal_price = float(df["price_nis"].quantile(price_q))
    rating_q = QUALITY_RATING_QUANTILE.get(params["quality_preference"], 0.55)
    ideal_rating = float(df["rating"].quantile(rating_q))
    ideal = pd.DataFrame([[ideal_price, ideal_rating]], columns=FEATURE_COLS)
    target_cluster = int(model.predict(scaler.transform(ideal))[0])

    # Labels computed on the full dataset so price tiers are stable, not relative to the slice.
    labels = assign_cluster_labels(model_result, df_full)
    df["cluster_label"] = df["cluster"].map(labels)

    # Persona-weighted ordering — used ONLY to rank within the model's chosen cluster.
    w = PERSONA_WEIGHTS.get(params["user_type"], PERSONA_WEIGHTS["student"])
    df["score"] = (
        w["rating"] * df["rating"]
        + w["price_nis"] * df["price_nis"] / 10
        + w["distance_km"] * df["distance_km"]
    )

    in_cluster = df[df["cluster"] == target_cluster].sort_values("score", ascending=False)
    # If the model's cluster has fewer than top_n venues, backfill with the next-best
    # candidates (still real venues, still ordered by the persona score).
    if len(in_cluster) < top_n:
        rest = df[df["cluster"] != target_cluster].sort_values("score", ascending=False)
        result = pd.concat([in_cluster, rest])
    else:
        result = in_cluster

    result = result.head(top_n).reset_index(drop=True)
    result.attrs["model_target_cluster"] = target_cluster
    result.attrs["model_target_label"] = labels.get(target_cluster, str(target_cluster))
    result.attrs["model_cluster_n"] = int((df["cluster"] == target_cluster).sum())
    return result


def fallback_recommend(
    df: pd.DataFrame,
    city: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Step 3 fallback: when the LLM output is unusable we ask only for the city and
    return three safe picks — best (איכות), cheapest (סכום), closest (מיקום)."""
    pool = df.dropna(subset=["price_nis", "rating", "weighted_rating"]).copy()
    pool = pool[pool["city"] == city]
    if pool.empty:
        return {}

    # If no GPS, measure "closest" against the city's own centroid as a sensible proxy.
    lat = user_lat if user_lat is not None else float(pool["lat"].mean())
    lng = user_lng if user_lng is not None else float(pool["lng"].mean())
    pool = _with_distance(pool, lat, lng)

    return {
        "best": pool.sort_values("rating", ascending=False).head(1).reset_index(drop=True),
        "cheapest": pool.sort_values("price_nis", ascending=True).head(1).reset_index(drop=True),
        "closest": pool.sort_values("distance_km", ascending=True).head(1).reset_index(drop=True),
    }


def phrase_response(user_text: str, params: dict, df_top: pd.DataFrame, api_key: str) -> str | None:
    """Step 4 (optional): the LLM phrases the MODEL's output in natural language.

    The LLM is handed the EXACT venues the model already chose, with their real numbers, and may
    only restate them — it cannot invent, add, drop, or change a venue, price, or rating. This is
    the 'output translation' half of the loop: the number stays the model's, the wording is the
    LLM's. Returns the text, or None on failure (the caller still shows the venue table)."""
    if df_top is None or df_top.empty:
        return None

    picks = "\n".join(
        f"{i}. {row['name']} — ⭐{row['rating']:.1f}, ₪{row['price_nis']:.0f}"
        for i, (_, row) in enumerate(df_top.iterrows(), start=1)
    )
    target = df_top.attrs.get("model_target_label", "")

    system = (
        "You are the response layer of a shawarma recommender. The ML model has ALREADY chosen "
        "the venues below, with their real numbers. Write a short, warm reply (2-4 sentences) IN "
        "THE SAME LANGUAGE the user wrote in. Hard rules: use ONLY the venues, prices and ratings "
        "listed below — never invent, add, drop, or change a venue or a number, and never output "
        "more venues than given. You may mention the model's cluster."
    )
    user = (
        f'The user wrote:\n"""\n{user_text.strip()}\n"""\n\n'
        f"The model placed this profile in cluster: {target}.\n"
        f"The model's selected venues (use these exactly, in this order):\n{picks}\n\n"
        "Write the reply now — no preamble, no JSON, just the message to the user."
    )
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,  # a little warmth in wording; the facts are fixed by the model
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return text or None
    except Exception:
        return None
