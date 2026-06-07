"""Clustering model: train venue profiles and rank recommendations per user persona."""
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURE_COLS = ["price_nis", "rating", "distance_km", "ratings_count"]

PERSONA_WEIGHTS = {
    "student": {"price_nis": -1.5, "rating": 1.0, "distance_km": -0.8},
    "quality": {"price_nis": -0.5, "rating": 2.0, "distance_km": -0.5},
}


def _persona_score(df: pd.DataFrame, persona: str) -> pd.Series:
    weights = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["student"])
    return (
        weights["rating"] * df["rating"]
        + weights["price_nis"] * df["price_nis"] / 10
        + weights["distance_km"] * df["distance_km"]
    )


def train(df: pd.DataFrame, k: int = 5) -> tuple[KMeans, StandardScaler, float]:
    """Fit KMeans on venue features. Returns (model, scaler, silhouette_score)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].fillna(0))
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X)
    score = silhouette_score(X, model.labels_)
    return model, scaler, score


def evaluate_silhouette(model: KMeans, scaler: StandardScaler, df: pd.DataFrame) -> float:
    X = scaler.transform(df[FEATURE_COLS].fillna(0))
    labels = model.predict(X)
    return silhouette_score(X, labels)


def baseline_distance_ranking(df: pd.DataFrame, persona: str = "student", top_k: int = 10) -> pd.DataFrame:
    baseline = df.sort_values("distance_km").head(top_k).copy()
    baseline["baseline_score"] = _persona_score(baseline, persona)
    return baseline


def predict(
    model: KMeans,
    scaler: StandardScaler,
    df: pd.DataFrame,
    persona: str = "student",
    max_dist_km: float = 2.0,
) -> pd.DataFrame:
    """Assign clusters and return venues ranked by persona-weighted score."""
    df = df[df["distance_km"] <= max_dist_km].copy()
    if df.empty:
        return df

    X = scaler.transform(df[FEATURE_COLS].fillna(0))
    df["cluster"] = model.predict(X)

    weights = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["student"])
    df["score"] = (
        weights["rating"] * df["rating"]
        + weights["price_nis"] * df["price_nis"] / 10
        + weights["distance_km"] * df["distance_km"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
