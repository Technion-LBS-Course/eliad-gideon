"""Clustering models: KMeans and DBSCAN trained on venue features with train/test KPI reporting."""
from __future__ import annotations
import pickle
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
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


def compare_algorithms(df: pd.DataFrame) -> tuple[dict, dict]:
    """Split data 70/30, train both algorithms, return results for comparison."""
    df_train, df_test = split_data(df)
    return train_kmeans(df_train, df_test), train_dbscan(df_train, df_test)


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


def evaluate_silhouette(model: KMeans, scaler: StandardScaler, df: pd.DataFrame) -> float:
    X = scaler.transform(df[FEATURE_COLS].fillna(0))
    labels = model.predict(X)
    return silhouette_score(X, labels)


def baseline_distance_ranking(df: pd.DataFrame, persona: str = "student", top_k: int = 10) -> pd.DataFrame:
    baseline = df.sort_values("distance_km").head(top_k).copy()
    baseline["baseline_score"] = _persona_score(baseline, persona)
    return baseline


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
