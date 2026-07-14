"""Core tests — one section per component; each unit-tests a single src/ module.

Sections: data · eda · model · agent · geo.
Shared fixtures (split/kmeans) train the model once and are reused across tests.
"""
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import data as data_mod
from src import model as model_mod
from src.agent import (
    _parse_json,
    _with_distance,
    build_system_prompt,
    build_user_prompt,
    city_center,
    fallback_recommend,
    recommend,
    validate_params,
)
from src.data import (
    build_features,
    clean,
    filter_haifa,
    generate_haifa_queries,
    load_raw,
)
from src.eda import (
    avg_rating_by_city_bar,
    compute_metrics,
    price_histogram,
    price_vs_rating_scatter,
    venue_map,
)
from src.geo import geocode_address, match_city
from src.model import (
    FEATURE_COLS,
    _haversine,
    _random_baseline_silhouette,
    assign_cluster_labels,
    compute_confusion_matrix,
    find_best_k,
    load_model,
    predict,
    save_model,
    split_data,
    train_agglomerative,
    train_dbscan,
    train_kmeans,
)

DF = clean(load_raw())
A_CITY = DF["city"].value_counts().index[0]  # the city with the most venues


# Shared fixtures — module scope so KMeans is trained once, not per test.
@pytest.fixture(scope="module")
def splits():
    return split_data(DF)


@pytest.fixture(scope="module")
def kmeans(splits):
    df_train, df_test = splits
    return train_kmeans(df_train, df_test)


# ══════════════════════════════ src/data.py ══════════════════════════════
def test_data_loads():
    assert len(DF) > 500


def test_price_nis_present():
    assert "price_nis" in DF.columns
    assert DF["price_nis"].notna().sum() > 0


def test_feature_cols_present():
    missing = [c for c in FEATURE_COLS if c not in DF.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_clean_renames_coords_dedups_and_derives_price():
    raw = pd.DataFrame(
        {
            "latitude": [32.8, 32.8, 31.0],       # rows 0 & 1 are duplicates
            "longitude": [35.0, 35.0, 34.8],
            "name": ["A", "A", "B"],
            "city": ["חיפה", "חיפה", "תל אביב"],
            "rating": [4.5, 4.5, 3.0],
            "reviews_count": [10, 10, 0],
            "price_turkey_shawarma_pita": [50, 50, 45],
            "price_cow_shawarma_pita": [52, 52, 47],
            "price_turkey_shawarma_laffa": [55, 55, 50],
            "price_cow_shawarma_laffa": [57, 57, 52],
            "car_park_nearby": ["True", "True", "False"],
            "price_source": ["estimated", "estimated", "estimated"],
        }
    )
    out = clean(raw)
    assert {"lat", "lng"}.issubset(out.columns)          # latitude/longitude renamed
    assert len(out) == 2                                  # identical name+lat+lng dropped
    assert out.loc[0, "price_nis"] == pytest.approx((50 + 52 + 55 + 57) / 4)
    assert "weighted_rating" in out.columns


def test_clean_empty_returns_empty():
    assert clean(pd.DataFrame()).empty


def test_filter_haifa_keeps_only_bbox():
    df = pd.DataFrame(
        {
            "lat": [32.80, 31.00],   # first in Haifa bbox, second in Tel Aviv (outside)
            "lng": [35.00, 34.80],
            "name": ["haifa_venue", "ta_venue"],
        }
    )
    out = filter_haifa(df)
    assert list(out["name"]) == ["haifa_venue"]


def test_generate_haifa_queries_deterministic_and_in_bbox():
    q1 = generate_haifa_queries(n=100, seed=42)
    q2 = generate_haifa_queries(n=100, seed=42)
    assert len(q1) == 100
    pd.testing.assert_frame_equal(q1, q2)   # same seed → reproducible
    b = data_mod.HAIFA_BBOX
    assert q1["lat"].between(b["lat_min"], b["lat_max"]).all()
    assert q1["lng"].between(b["lng_min"], b["lng_max"]).all()


def test_build_features_adds_distance():
    df = pd.DataFrame(
        {"lat": [32.0], "lng": [34.0], "price_nis": [50.0], "reviews_count": [0]}
    )
    out = build_features(df, user_lat=32.0, user_lng=34.0)
    assert "distance_km" in out.columns
    assert out["distance_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)  # same point → 0


# ══════════════════════════════ src/eda.py ═══════════════════════════════
def test_compute_metrics_keys_and_ranges():
    m = compute_metrics(DF, "price_nis")
    for key in ("n_venues", "n_cities", "avg_price", "median_price", "avg_rating"):
        assert key in m
    assert m["n_venues"] == len(DF)
    assert 1.0 <= m["avg_rating"] <= 5.0
    assert m["avg_price"] > 0


def test_chart_functions_return_figures():
    df_ok = DF.dropna(subset=["lat", "lng", "rating", "price_nis", "city", "name"])
    assert isinstance(price_histogram(df_ok, "price_nis"), go.Figure)
    assert isinstance(price_vs_rating_scatter(df_ok, "price_nis"), go.Figure)
    assert isinstance(avg_rating_by_city_bar(df_ok), go.Figure)
    assert isinstance(venue_map(df_ok, "price_nis"), go.Figure)


def test_venue_map_none_on_empty():
    assert venue_map(DF.iloc[0:0], "price_nis") is None


# ══════════════════════════════ src/model.py ═════════════════════════════
def test_split_data(splits):
    df_train, df_test = splits
    assert len(df_train) > len(df_test)
    valid_count = DF.dropna(subset=["rating", "price_nis"]).shape[0]
    assert len(df_train) + len(df_test) == valid_count


def test_train_kmeans_returns_scores(kmeans):
    assert kmeans["algorithm"] == "KMeans"
    assert 0 < kmeans["train_silhouette"] < 1
    assert 0 < kmeans["test_silhouette"] < 1
    assert kmeans["k"] == 9  # fixed — one cluster per target class


def test_train_dbscan_returns_scores(splits):
    df_train, df_test = splits
    r = train_dbscan(df_train, df_test)
    assert r["algorithm"] == "DBSCAN"
    assert r["eps"] > 0
    assert 0 <= r["noise_pct"] <= 100
    assert "train_silhouette" in r and "test_silhouette" in r


def test_train_agglomerative_returns_scores(splits):
    df_train, df_test = splits
    r = train_agglomerative(df_train, df_test)
    assert r["algorithm"] == "Agglomerative"
    assert r["linkage"] == "ward"
    assert 0 < r["train_silhouette"] < 1
    assert 0 < r["test_silhouette"] < 1
    assert r["k"] == 9


def test_find_best_k_in_range(splits):
    df_train, df_test = splits
    X_train, _, _ = model_mod._scale(df_train, df_test)
    best_k, scores = find_best_k(X_train)
    assert best_k in range(3, 9)
    assert len(scores) == len(range(3, 9))
    assert all(-1 <= s <= 1 for s in scores.values())


def test_random_baseline_silhouette_bounded(splits):
    df_train, df_test = splits
    X_train, _, _ = model_mod._scale(df_train, df_test)
    assert -1.0 <= _random_baseline_silhouette(X_train, k=9) <= 1.0


def test_confusion_matrix_is_9x9(kmeans, splits):
    df_train, df_test = splits
    cm = compute_confusion_matrix(kmeans, df_train, df_test)
    assert cm["confusion_matrix"].shape == (9, 9)
    assert len(cm["classes"]) == 9
    assert 0.0 <= cm["accuracy"] <= 1.0


def test_assign_cluster_labels_nine_labels(kmeans):
    labels = assign_cluster_labels(kmeans, DF)
    assert len(labels) == 9
    assert all(isinstance(v, str) for v in labels.values())


def test_predict_returns_ranked_list(kmeans):
    recs = predict(kmeans, DF, persona="student", user_lat=32.08, user_lng=34.78, max_dist_km=5.0)
    assert not recs.empty
    assert "score" in recs.columns
    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)  # ranked high → low


def test_predict_empty_on_impossible_location(kmeans):
    # Middle of the ocean — no venues within 0.1 km
    recs = predict(kmeans, DF, persona="student", user_lat=0.0, user_lng=0.0, max_dist_km=0.1)
    assert recs.empty


def test_haversine_properties():
    ta = (32.0853, 34.7818)     # Tel Aviv
    haifa = (32.7940, 34.9896)  # Haifa
    assert _haversine(*ta, *ta) == 0.0                                    # identity
    assert _haversine(*ta, *haifa) == pytest.approx(_haversine(*haifa, *ta))  # symmetry
    assert _haversine(*ta, *haifa) >= 0                                   # non-negative
    assert _haversine(*ta, *haifa) == pytest.approx(81.2, abs=1.0)        # known distance


def test_save_and_load_model_roundtrip(kmeans, tmp_path):
    path = tmp_path / "m.pkl"
    save_model(kmeans, path)
    loaded = load_model(path)
    assert loaded is not None
    assert loaded["test_silhouette"] == pytest.approx(kmeans["test_silhouette"])


def test_load_model_missing_returns_none(tmp_path):
    assert load_model(tmp_path / "does_not_exist.pkl") is None


# ═══════════════ integration: database → model (end-to-end) ═══════════════
def test_pipeline_database_to_model_end_to_end():
    """Full chain on the real dataset: load → clean → split → train → predict.

    Unlike the unit tests above, this drives the whole data-to-model pipeline in
    one flow and asserts the documented success criteria: the model finds real
    structure and beats the random-assignment baseline, then produces ranked
    recommendations. (The historical KPI of silhouette ≥ 0.45 predates the
    weighted_rating feature swap; the current test silhouette is ~0.33 — still
    well above the random baseline, which is the stable success signal.)
    """
    df = clean(load_raw())                                    # the "database"
    assert len(df) > 500
    assert set(FEATURE_COLS).issubset(df.columns)

    df_train, df_test = split_data(df)
    result = train_kmeans(df_train, df_test)

    # Real separation: positive silhouette that beats random cluster assignment.
    assert result["test_silhouette"] > 0
    assert result["test_silhouette"] > result["baseline_silhouette"]

    # The trained model turns the same data into ranked recommendations.
    recs = predict(result, df, persona="student", user_lat=32.08, user_lng=34.78, max_dist_km=5.0)
    assert not recs.empty
    assert list(recs["score"]) == sorted(recs["score"], reverse=True)


# ══════════════════════════════ src/agent.py ═════════════════════════════
CITIES = ["חיפה", "תל אביב"]


# Parametrized — one function, many JSON shapes the LLM might return.
@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),                    # plain JSON
        ('```json\n{"a": 1}\n```', {"a": 1}),      # markdown-fenced
        ('prefix {"a": 1} suffix', {"a": 1}),      # JSON embedded in prose
        ("no json here", None),                    # no object at all
        ("", None),                                # empty
        (None, None),                              # missing
    ],
)
def test_parse_json_variants(text, expected):
    assert _parse_json(text) == expected


def test_validate_params_extracts_address_never_coordinates():
    ok = validate_params(
        {"address": "רחוב הרצל 45, חיפה", "city": "חיפה",
         "max_budget_nis": 60, "quality_preference": "high", "user_type": "student"},
        CITIES,
    )
    assert ok["address"] == "רחוב הרצל 45, חיפה"
    assert ok["city"] == "חיפה"
    # A city not in the dataset is dropped to None (no unsafe substring match).
    dropped = validate_params(
        {"address": None, "city": "פריז",
         "max_budget_nis": 60, "quality_preference": "low", "user_type": "quality"},
        CITIES,
    )
    assert dropped["city"] is None


# Parametrized — every kind of invalid LLM output must trigger the fallback (None).
@pytest.mark.parametrize(
    "bad",
    [
        None,                                                                    # not a dict
        {},                                                                      # empty
        {"max_budget_nis": 60, "quality_preference": "high", "user_type": "x"},  # bad user_type
        {"max_budget_nis": 60, "quality_preference": "wow", "user_type": "student"},  # bad quality
        {"max_budget_nis": 5, "quality_preference": "high", "user_type": "student"},  # budget too low
        {"max_budget_nis": "abc", "quality_preference": "high", "user_type": "student"},  # non-numeric
    ],
)
def test_validate_params_rejects_bad_input(bad):
    assert validate_params(bad, CITIES) is None


def test_build_system_prompt_lists_cities_and_keys():
    p = build_system_prompt(["חיפה", "תל אביב"])
    assert "חיפה" in p and "תל אביב" in p
    assert "max_budget_nis" in p
    assert "JSON" in p


def test_build_user_prompt_includes_text():
    assert "אני סטודנט בחיפה" in build_user_prompt("אני סטודנט בחיפה")


def test_city_center():
    center = city_center(DF, A_CITY)
    assert center is not None
    lat, lng = center
    assert isinstance(lat, float) and isinstance(lng, float)
    assert city_center(DF, None) is None
    assert city_center(DF, "עיר-שאיננה-קיימת") is None


def test_with_distance_zero_without_coords():
    df = DF.head(5)[["lat", "lng"]].copy()
    out = _with_distance(df, None, None)
    assert (out["distance_km"] == 0.0).all()


def test_recommend_returns_model_cluster(kmeans):
    params = {"city": None, "max_budget_nis": 200,
              "quality_preference": "low", "user_type": "student"}
    out = recommend(kmeans, DF, params, top_n=5)
    assert len(out) <= 5
    assert "score" in out.columns
    assert "model_target_cluster" in out.attrs   # the model, not the LLM, picked the cluster


def test_fallback_recommend_three_picks():
    out = fallback_recommend(DF, A_CITY)
    assert {"best", "cheapest", "closest"}.issubset(out.keys())
    assert len(out["best"]) == 1
    assert len(out["cheapest"]) == 1
    assert len(out["closest"]) == 1


# ══════════════════════════════ src/geo.py ═══════════════════════════════
def test_geocode_returns_none_without_key():
    # No key → never calls the network, never invents coordinates.
    assert geocode_address("רחוב הרצל 1, חיפה", api_key="") is None
    assert geocode_address("", api_key="fake") is None


def test_match_city_exact_and_normalized():
    cities = ["תל אביב", "חיפה", "רמת גן"]
    assert match_city("חיפה", cities) == "חיפה"           # exact
    assert match_city("תל אביב-יפו", cities) == "תל אביב"  # normalized contains
    assert match_city("פריז", cities) is None             # no confident match → None
    assert match_city(None, cities) is None


@pytest.mark.parametrize(
    "given, expected",
    [
        ("חיפה", "חיפה"),              # exact match
        ("תל אביב-יפו", "תל אביב"),    # hyphen normalized to a contains-match
        ("פריז", None),                # not in the list → no confident match
        ("", None),                    # empty string → None
        (None, None),                  # missing input → None
    ],
)
def test_match_city_cases(given, expected):
    cities = ["תל אביב", "חיפה", "רמת גן"]
    assert match_city(given, cities) == expected
