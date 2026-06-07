"""Smoke tests — must pass on every commit."""
import pandas as pd
import pytest
from src.data import clean, load_raw
from src.model import (
    FEATURE_COLS,
    predict,
    split_data,
    train_agglomerative,
    train_kmeans,
)

DF = clean(load_raw())


def test_data_loads():
    assert len(DF) > 10_000


def test_price_nis_present():
    assert "price_nis" in DF.columns
    assert DF["price_nis"].notna().sum() > 0


def test_feature_cols_present():
    missing = [c for c in FEATURE_COLS if c not in DF.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_split_data():
    df_train, df_test = split_data(DF)
    assert len(df_train) > len(df_test)
    # Combined size equals valid rows (those with price_nis + rating)
    valid_count = DF.dropna(subset=["rating", "price_nis"]).shape[0]
    assert len(df_train) + len(df_test) == valid_count


def test_train_kmeans_returns_scores():
    df_train, df_test = split_data(DF)
    result = train_kmeans(df_train, df_test)
    assert result["algorithm"] == "KMeans"
    assert 0 < result["train_silhouette"] < 1
    assert 0 < result["test_silhouette"] < 1
    assert 3 <= result["k"] <= 8


def test_predict_returns_ranked_list():
    df_train, df_test = split_data(DF)
    result = train_kmeans(df_train, df_test)
    recs = predict(result, DF, persona="student", user_lat=32.08, user_lng=34.78, max_dist_km=5.0)
    assert not recs.empty
    assert "score" in recs.columns
    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_train_agglomerative_returns_scores():
    df_train, df_test = split_data(DF)
    result = train_agglomerative(df_train, df_test)
    assert result["algorithm"] == "Agglomerative"
    assert result["linkage"] == "ward"
    assert 0 < result["train_silhouette"] < 1
    assert 0 < result["test_silhouette"] < 1
    assert 3 <= result["k"] <= 8


def test_predict_empty_on_impossible_location():
    df_train, df_test = split_data(DF)
    result = train_kmeans(df_train, df_test)
    # Middle of the ocean — no venues within 0.1 km
    recs = predict(result, DF, persona="student", user_lat=0.0, user_lng=0.0, max_dist_km=0.1)
    assert recs.empty
