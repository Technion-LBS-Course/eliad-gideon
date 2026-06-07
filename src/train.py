from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import build_features, clean, load_raw
from src.model import (
    baseline_distance_ranking,
    evaluate_silhouette,
    FEATURE_COLS,
    train,
    _persona_score,
    predict,
)

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

DEFAULT_LAT = 31.5
DEFAULT_LNG = 34.9


def save_artifact(path: Path, artifact: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(artifact, handle)


def load_artifact(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def prepare_training_data() -> pd.DataFrame:
    df = clean(load_raw())
    return build_features(df, DEFAULT_LAT, DEFAULT_LNG)


def compare_to_baseline(df_test: pd.DataFrame, persona: str = "student") -> dict[str, float]:
    baseline = baseline_distance_ranking(df_test, persona=persona, top_k=10)
    model_top = predict(
        *train(df_test, k=5)[:2],
        df=df_test,
        persona=persona,
        max_dist_km=df_test["distance_km"].max(),
    ).head(10)

    return {
        "baseline_top10_score": float(baseline["baseline_score"].mean()),
        "model_top10_score": float(model_top["score"].mean()),
    }


def run_k_sweep(df_train: pd.DataFrame, k_values: range) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for k in k_values:
        _, _, score = train(df_train, k=k)
        results.append({"k": k, "train_silhouette": float(score)})
    return results


def main() -> int:
    df_features = prepare_training_data()
    df_train, df_test = train_test_split(df_features, test_size=0.15, random_state=42)

    print(f"Training venues: {len(df_train):,}")
    print(f"Test venues: {len(df_test):,}\n")

    sweep_results = run_k_sweep(df_train, range(3, 9))
    for row in sweep_results:
        print(f"k={row['k']} -> train silhouette={row['train_silhouette']:.4f}")

    best_k = max(sweep_results, key=lambda row: row["train_silhouette"])["k"]
    model, scaler, train_score = train(df_train, k=best_k)
    test_score = evaluate_silhouette(model, scaler, df_test)

    save_artifact(MODEL_PATH, model)
    save_artifact(SCALER_PATH, scaler)

    print("\nBest model:\n")
    print(f"  k = {best_k}")
    print(f"  train silhouette = {train_score:.4f}")
    print(f"  test silhouette  = {test_score:.4f}\n")

    for persona in ["student", "quality"]:
        baseline = baseline_distance_ranking(df_test, persona=persona, top_k=10)
        model_top = predict(model, scaler, df_test, persona=persona, max_dist_km=df_test["distance_km"].max()).head(10)
        print(f"Persona: {persona}")
        print(f"  baseline avg persona score top 10 = {baseline['baseline_score'].mean():.4f}")
        print(f"  model avg persona score top 10    = {model_top['score'].mean():.4f}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
