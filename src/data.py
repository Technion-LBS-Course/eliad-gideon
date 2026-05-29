from pathlib import Path
import pandas as pd
from math import radians, cos, sin, asin, sqrt

ROOT = Path(__file__).parent.parent
_DEFAULT_CSV = ROOT / "israel_shawarma_database_full_prices_filled.csv"

PRICE_COLS = [
    "price_turkey_shawarma_pita",
    "price_cow_shawarma_pita",
    "price_turkey_shawarma_laffa",
    "price_cow_shawarma_laffa",
    "price_falafel_pita",
    "price_falafel_laffa",
]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or _DEFAULT_CSV
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # Normalise coordinate column names to lat/lng for consistency with build_features
    df = df.rename(columns={"latitude": "lat", "longitude": "lng"})

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["reviews_count"] = pd.to_numeric(df["reviews_count"], errors="coerce")
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["car_park_nearby"] = df["car_park_nearby"].map({"True": True, "False": False})

    df = df.dropna(subset=["lat", "lng"])
    df = df.drop_duplicates(subset=["name", "lat", "lng"])
    return df.reset_index(drop=True)


def build_features(df_clean: pd.DataFrame, user_lat: float, user_lng: float) -> pd.DataFrame:
    def haversine(lat1, lng1, lat2, lng2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlng = radians(lng2 - lng1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
        return 2 * R * asin(sqrt(a))

    df = df_clean.copy()
    df["distance_km"] = df.apply(
        lambda r: haversine(user_lat, user_lng, r["lat"], r["lng"]), axis=1
    )
    return df
