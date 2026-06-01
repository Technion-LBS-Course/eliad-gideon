from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


PRICE_COLS = [
    "price_turkey_shawarma_pita",
    "price_cow_shawarma_pita",
    "price_turkey_shawarma_laffa",
    "price_cow_shawarma_laffa",
    "price_falafel_pita",
    "price_falafel_laffa",
]


def compute_metrics(df: pd.DataFrame, price_col: str) -> dict:
    base = df.dropna(subset=[price_col])
    rated = base.dropna(subset=["rating"])
    return {
        "n_venues": len(df),
        "n_cities": int(df["city"].nunique()),
        "avg_price": float(base[price_col].mean()),
        "median_price": float(base[price_col].median()),
        "avg_rating": float(rated["rating"].mean()),
        "null_ratings": int(df["rating"].isna().sum()),
        "estimated_prices": int((df["price_source"] == "estimated").sum()),
        "unique_venues": int(df["name"].nunique()),
    }


def price_histogram(df: pd.DataFrame, price_col: str) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=[price_col]),
        x=price_col,
        nbins=30,
        color_discrete_sequence=["#f4a261"],
        labels={price_col: "Price (NIS)", "count": "Venues"},
    )
    fig.update_layout(showlegend=False, bargap=0.05, yaxis_title="Venues")
    return fig


def price_vs_rating_scatter(df: pd.DataFrame, price_col: str) -> go.Figure:
    scatter_df = df[[price_col, "rating", "city", "name"]].dropna()
    coeffs = np.polyfit(scatter_df[price_col], scatter_df["rating"], 1)
    x_line = np.linspace(scatter_df[price_col].min(), scatter_df[price_col].max(), 100)
    y_line = np.polyval(coeffs, x_line)
    corr = scatter_df[price_col].corr(scatter_df["rating"])

    fig = px.scatter(
        scatter_df,
        x=price_col,
        y="rating",
        color="city",
        hover_name="name",
        opacity=0.35,
        labels={price_col: "Price (NIS)", "rating": "Rating"},
        height=420,
    )
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#e63946", width=2, dash="dash"),
        name=f"Trend (r={corr:.2f})",
        showlegend=True,
    ))
    fig.update_traces(marker_size=4, selector=dict(mode="markers"))
    return fig


def avg_rating_by_city_bar(df: pd.DataFrame, min_venues: int = 5, top_n: int = 20) -> go.Figure:
    city_rating = (
        df.dropna(subset=["rating"])
        .groupby("city")["rating"]
        .agg(["mean", "count"])
        .query(f"count >= {min_venues}")
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "venues"})
    )
    fig = px.bar(
        city_rating,
        x="avg_rating",
        y="city",
        orientation="h",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        range_color=[3.5, 5.0],
        hover_data={"venues": True},
        labels={"avg_rating": "Avg Rating", "city": ""},
        height=420,
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def venue_map(df: pd.DataFrame, price_col: str) -> go.Figure | None:
    df_map = df.dropna(subset=["lat", "lng", "rating"])
    if df_map.empty:
        return None
    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lng",
        color="rating",
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        size=price_col,
        size_max=10,
        hover_name="name",
        hover_data={"city": True, price_col: True, "rating": True, "lat": False, "lng": False},
        zoom=7,
        center={"lat": 31.5, "lon": 34.9},
        height=520,
        mapbox_style="open-street-map",
        opacity=0.75,
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Rating", thickness=12),
    )
    return fig
