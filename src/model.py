"""Clustering models: KMeans, DBSCAN, Agglomerative — train/test KPI reporting."""
from __future__ import annotations
import pickle
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["price_nis", "rating", "reviews_count"]
MODEL_PATH = Path(__file__).parent.parent / "data" / "kmeans_model.pkl"

PERSONA_WEIGHTS = {
    "student": {"price_nis": -1.5, "rating": 1.0, "distance_km": -0.8},
    "quality": {"price_nis": -0.5, "rating": 2.0, "distance_km": -0.5},
}


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 stratified train/test split on venues with valid price and rating."""
    df_valid = df.dropna(subset=["rating", "price_nis"]).copy()
    df_train, df_test = train_test_split(df_valid, test_size=0.30, random_state=42)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def _scale(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].fillna(0))
    X_test = scaler.transform(df_test[FEATURE_COLS].fillna(0))
    return X_train, X_test, scaler


def find_best_k(X_train: np.ndarray, k_range=range(3, 9)) -> tuple[int, dict[int, float]]:
    """Sweep k values; return best k by silhouette and per-k scores for elbow chart."""
    scores: dict[int, float] = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X_train)
        scores[k] = float(silhouette_score(X_train, labels))
    best_k = max(scores, key=scores.__getitem__)
    return best_k, scores


def train_kmeans(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train KMeans with auto-tuned k; return model artifacts and train/test silhouette."""
    X_train, X_test, scaler = _scale(df_train, df_test)
    best_k, k_scores = find_best_k(X_train)

    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    model.fit(X_train)

    train_sil = float(silhouette_score(X_train, model.labels_))
    test_sil = float(silhouette_score(X_test, model.predict(X_test)))

    return {
        "algorithm": "KMeans",
        "model": model,
        "scaler": scaler,
        "k": best_k,
        "k_scores": k_scores,
        "train_silhouette": train_sil,
        "test_silhouette": test_sil,
    }


def train_dbscan(
    df_train: pd.DataFrame, df_test: pd.DataFrame, eps: float = 0.5, min_samples: int = 5
) -> dict:
    """Train DBSCAN; assign test labels via KNN on core samples for out-of-sample KPI."""
    X_train, X_test, scaler = _scale(df_train, df_test)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    train_labels = model.fit_predict(X_train)

    mask_train = train_labels != -1
    noise_pct = float((~mask_train).mean() * 100)

    if mask_train.sum() > 1:
        train_sil = float(silhouette_score(X_train[mask_train], train_labels[mask_train]))
    else:
        train_sil = 0.0

    # Assign test points to nearest core sample cluster for out-of-sample evaluation
    core_idx = model.core_sample_indices_
    if len(core_idx) > 0:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train[core_idx], train_labels[core_idx])
        test_labels = knn.predict(X_test)
        test_sil = float(silhouette_score(X_test, test_labels)) if len(np.unique(test_labels)) > 1 else 0.0
    else:
        test_sil = 0.0

    return {
        "algorithm": "DBSCAN",
        "model": model,
        "scaler": scaler,
        "eps": eps,
        "min_samples": min_samples,
        "noise_pct": noise_pct,
        "train_silhouette": train_sil,
        "test_silhouette": test_sil,
    }


def train_agglomerative(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train AgglomerativeClustering (ward linkage) with auto-tuned k.
    Test labels assigned via 1-NN on training set (no native predict support)."""
    X_train, X_test, scaler = _scale(df_train, df_test)

    # Reuse the same k sweep as KMeans for a fair comparison
    best_k, k_scores = find_best_k(X_train)

    model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    train_labels = model.fit_predict(X_train)
    train_sil = float(silhouette_score(X_train, train_labels))

    # 1-NN assignment for out-of-sample test evaluation
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, train_labels)
    test_labels = knn.predict(X_test)
    test_sil = float(silhouette_score(X_test, test_labels))

    return {
        "algorithm": "Agglomerative",
        "model": model,
        "knn": knn,           # kept for predict() fallback
        "scaler": scaler,
        "k": best_k,
        "linkage": "ward",
        "train_silhouette": train_sil,
        "test_silhouette": test_sil,
    }


def compare_algorithms(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Split data 70/30, train all three algorithms, return results for comparison."""
    df_train, df_test = split_data(df)
    return (
        train_kmeans(df_train, df_test),
        train_dbscan(df_train, df_test),
        train_agglomerative(df_train, df_test),
    )


def assign_cluster_labels(result: dict, df: pd.DataFrame) -> dict[int, str]:
    """Map each cluster ID to a semantic label based on its centroid price and rating.

    Uses dataset-wide 33rd/67th percentiles as thresholds so labels are relative
    to the actual distribution rather than arbitrary absolute values.
    """
    model = result["model"]
    scaler = result["scaler"]

    centroids = scaler.inverse_transform(model.cluster_centers_)
    price_idx = FEATURE_COLS.index("price_nis")
    rating_idx = FEATURE_COLS.index("rating")

    price_33 = float(df["price_nis"].quantile(0.33))
    price_67 = float(df["price_nis"].quantile(0.67))
    rating_33 = float(df["rating"].quantile(0.33))
    rating_67 = float(df["rating"].quantile(0.67))

    def _rating_label(r: float) -> str:
        if r >= rating_67:
            return "Good"
        if r >= rating_33:
            return "Average"
        return "Below Average"

    def _price_label(p: float) -> str:
        if p >= price_67:
            return "Expensive"
        if p >= price_33:
            return "Reasonable"
        return "Affordable"

    labels: dict[int, str] = {}
    seen: dict[str, int] = {}
    for cid, centroid in enumerate(centroids):
        base = f"{_rating_label(centroid[rating_idx])} & {_price_label(centroid[price_idx])}"
        # Disambiguate duplicates with the centroid price
        label = base if base not in seen else f"{base} (₪{centroid[price_idx]:.0f})"
        labels[cid] = label
        seen[base] = cid

    return labels


def save_model(result: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * R * asin(sqrt(a))


def predict(
    result: dict,
    df: pd.DataFrame,
    persona: str = "student",
    user_lat: float = 31.0,
    user_lng: float = 34.8,
    max_dist_km: float = 2.0,
) -> pd.DataFrame:
    """Filter venues by distance, assign clusters, rank by persona-weighted score."""
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: _haversine(user_lat, user_lng, r["lat"], r["lng"]), axis=1
    )
    df = df[df["distance_km"] <= max_dist_km].dropna(subset=["price_nis", "rating"])
    if df.empty:
        return df

    X = result["scaler"].transform(df[FEATURE_COLS].fillna(0))
    df["cluster"] = result["model"].predict(X)

    w = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["student"])
    df["score"] = (
        w["rating"] * df["rating"]
        + w["price_nis"] * df["price_nis"] / 10
        + w["distance_km"] * df["distance_km"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── Haifa-specific model (includes lat/lng as clustering features) ──────────

HAIFA_FEATURE_COLS = ["price_nis", "rating", "lat", "lng"]

# Geographic sub-areas within the bounding box
def _haifa_area(lat: float) -> str:
    if lat >= 32.84:
        return "Krayot"
    if lat >= 32.78:
        return "Haifa"
    return "Carmel"


def train_haifa_model(df: pd.DataFrame) -> dict:
    """Train KMeans on Haifa venues using [price_nis, rating, lat, lng] features.
    Including coordinates lets the model discover clusters that are both
    geographically and quality-cohesive (e.g. 'Good & Affordable in Carmel')."""
    df_valid = df.dropna(subset=HAIFA_FEATURE_COLS).copy()
    df_train, df_test = train_test_split(df_valid, test_size=0.30, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[HAIFA_FEATURE_COLS])
    X_test = scaler.transform(df_test[HAIFA_FEATURE_COLS])

    best_k, k_scores = find_best_k(X_train)

    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    model.fit(X_train)

    train_sil = float(silhouette_score(X_train, model.labels_))
    test_sil = float(silhouette_score(X_test, model.predict(X_test)))

    return {
        "algorithm": "KMeans-Haifa",
        "model": model,
        "scaler": scaler,
        "feature_cols": HAIFA_FEATURE_COLS,
        "k": best_k,
        "k_scores": k_scores,
        "train_silhouette": train_sil,
        "test_silhouette": test_sil,
    }


def assign_haifa_cluster_labels(result: dict, df: pd.DataFrame) -> dict[int, str]:
    """Semantic labels for Haifa clusters: quality × price × geographic sub-area."""
    model = result["model"]
    scaler = result["scaler"]
    feature_cols = result.get("feature_cols", HAIFA_FEATURE_COLS)

    centroids = scaler.inverse_transform(model.cluster_centers_)
    price_idx = feature_cols.index("price_nis")
    rating_idx = feature_cols.index("rating")
    lat_idx = feature_cols.index("lat")

    price_33 = float(df["price_nis"].quantile(0.33))
    price_67 = float(df["price_nis"].quantile(0.67))
    rating_33 = float(df["rating"].quantile(0.33))
    rating_67 = float(df["rating"].quantile(0.67))

    def _rl(r: float) -> str:
        return "Good" if r >= rating_67 else ("Average" if r >= rating_33 else "Below Average")

    def _pl(p: float) -> str:
        return "Expensive" if p >= price_67 else ("Reasonable" if p >= price_33 else "Affordable")

    labels: dict[int, str] = {}
    seen: dict[str, int] = {}
    for cid, c in enumerate(centroids):
        base = f"{_rl(c[rating_idx])} & {_pl(c[price_idx])} — {_haifa_area(c[lat_idx])}"
        label = base if base not in seen else f"{base} (₪{c[price_idx]:.0f})"
        labels[cid] = label
        seen[base] = cid

    return labels


def predict_from_map(
    result: dict,
    df: pd.DataFrame,
    user_lat: float,
    user_lng: float,
    persona: str = "student",
    max_dist_km: float = 2.0,
) -> pd.DataFrame:
    """Filter Haifa venues by distance from map pin, cluster, and rank by persona score."""
    feature_cols = result.get("feature_cols", HAIFA_FEATURE_COLS)
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: _haversine(user_lat, user_lng, r["lat"], r["lng"]), axis=1
    )
    df = df[df["distance_km"] <= max_dist_km].dropna(subset=["price_nis", "rating"])
    if df.empty:
        return df

    X = result["scaler"].transform(df[feature_cols].fillna(0))
    df["cluster"] = result["model"].predict(X)

    w = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["student"])
    df["score"] = (
        w["rating"] * df["rating"]
        + w["price_nis"] * df["price_nis"] / 10
        + w["distance_km"] * df["distance_km"]
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
