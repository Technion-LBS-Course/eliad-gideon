import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.data import build_features, load_raw, clean
from src.model import predict, train

st.set_page_config(page_title="Appetite Engineering", layout="wide", page_icon="🥙")

PRICE_LABELS = {
    "Turkey Shawarma – Pita":  "price_turkey_shawarma_pita",
    "Cow Shawarma – Pita":     "price_cow_shawarma_pita",
    "Turkey Shawarma – Laffa": "price_turkey_shawarma_laffa",
    "Cow Shawarma – Laffa":    "price_cow_shawarma_laffa",
    "Falafel – Pita":          "price_falafel_pita",
    "Falafel – Laffa":         "price_falafel_laffa",
}


@st.cache_data(show_spinner="Loading venue data…")
def get_data() -> pd.DataFrame:
    return clean(load_raw())


df_all = get_data()

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
DEFAULT_LAT = 31.5
DEFAULT_LNG = 34.9

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH, "rb") as handle:
            model = pickle.load(handle)
        with open(SCALER_PATH, "rb") as handle:
            scaler = pickle.load(handle)
        return model, scaler

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, scaler, _ = train(build_features(df_all, DEFAULT_LAT, DEFAULT_LNG), k=5)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)
    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)
    return model, scaler

# ── Sidebar — applies to EDA tab ───────────────────────────────
with st.sidebar:
    st.header("EDA Filters")
    product_label = st.selectbox("Product", list(PRICE_LABELS.keys()))
    price_col = PRICE_LABELS[product_label]
    cities = sorted(df_all["city"].dropna().unique().tolist())
    selected_cities = st.multiselect("Cities", cities)
    price_min = int(df_all[price_col].min())
    price_max = int(df_all[price_col].max())
    sel_price = st.slider("Price range (NIS)", price_min, price_max, (price_min, price_max))
    sel_rating = st.slider("Min rating", 1.0, 5.0, 1.0, step=0.1)
    only_parking = st.checkbox("Parking available only")

# Filtered slice for EDA tab
df = df_all.copy()
if selected_cities:
    df = df[df["city"].isin(selected_cities)]
df = df[df[price_col].between(*sel_price)]
df_rated = df.dropna(subset=["rating"])
df_rated = df_rated[df_rated["rating"] >= sel_rating]
if only_parking:
    df_rated = df_rated[df_rated["car_park_nearby"] == True]

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Problem & Personas",
    "📚 Literature & Market",
    "📊 EDA",
    "🏆 KPI & Model",
    "🎯 Recommendation",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Problem & Personas
# ══════════════════════════════════════════════════════════════
with tab1:
    st.title("Appetite Engineering — הנדסת התיאבון")

    st.info(
        "**Value Proposition:** ML-powered shawarma navigation that clusters 12,000+ Israeli venues "
        "by price, rating, and distance — then ranks them in real time to match your personal hunger profile."
    )

    st.divider()

    # ── Personas ──────────────────────────────────────────────
    st.subheader("User Personas")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        with st.container(border=True):
            st.markdown("### 👑 Quality Enthusiast — חובב האיכות")
            st.markdown("""
**Profile:** 28–40 · urban professional · disposable income

**Goal:** The perfect shawarma experience — full culinary satisfaction, even if it costs 60 NIS and requires travel.

**Algorithm weights:** Rating ×2.0 · Price ×0.5 · Distance ×0.5

**Threshold:** Rating ≥ 4.8 only
            """)

    with col_p2:
        with st.container(border=True):
            st.markdown("### 🎓 Thrifty Student — הסטודנט החסכן")
            st.markdown("""
**Profile:** 20–26 · student · tight budget (45–52 NIS ceiling)

**Goal:** Maximize calories per shekel without sacrificing taste.

**Algorithm weights:** Price ×1.5 · Rating ×1.0 · Distance ×0.8

**Threshold:** Rating ≥ 4.0 · Price ≤ 52 NIS
            """)

    st.divider()

    # ── Before / After ────────────────────────────────────────
    st.subheader("Before / After — The Problem in Practice")
    col_b, col_a = st.columns(2)

    with col_b:
        st.error("""
**❌ BEFORE — Without Appetite Engineering**

It's 13:10. Eyal has 30 minutes between lectures.
Opens Google Maps → "shawarma near me" → 47 results, no exact price, no real ranking.
Spends 8 minutes scrolling. Picks the first 4.2⭐ place he recognises from memory.
Pays **58 NIS** for a 3.1⭐ shawarma.
Returns late to class.
Cognitive load: **high**. Satisfaction: **low**.
        """)

    with col_a:
        st.success("""
**✅ AFTER — With Appetite Engineering**

Same scenario. Eyal opens the app, selects **Student** persona.
GPS fires. In 3 seconds: top-3 venues within 800 m, sorted by value score.
**#1: 4.7⭐ · ₪47 · 6-minute walk.**
Eyal pays 11 NIS less and gets a better shawarma.
Returns on time.
Cognitive load: **zero**. Satisfaction: **high**.
        """)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Literature & Market Survey
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Literature Review")
    lit_data = {
        "Paper": [
            "Bondevik et al. (2024) — A systematic review on food recommender systems · Expert Systems With Applications 238, 122166",
            "Villegas et al. (2018) — Characterizing context-aware recommender systems · Knowledge-Based Systems 140, 173–200",
            "Asani et al. (2021) — Restaurant recommender system based on sentiment analysis · Machine Learning with Applications 6, 100114",
        ],
        "Method": [
            "Systematic literature review of 67 food RS (2017–2022, 2,738 papers screened); ML adoption taxonomy",
            "SLR of 87 context-aware RS; pre/post-filtering vs. contextual modeling framework; location + time as dominant context",
            "NLP + hierarchical clustering (Wu-Palmer) + SentiWordNet + cosine similarity; 92.8% Top-5 precision",
        ],
        "Relevance to Project": [
            "State-of-the-art landscape of food recommender systems — confirms our ML clustering approach is in the high-value minority",
            "Validates location + time as the two most impactful context dimensions; justifies distance_km as a core feature",
            "Demonstrates that clustering restaurant features achieves high recommendation precision vs. keyword-sort baselines",
        ],
        "Key Takeaway We Apply": [
            "64.79% of food RS use ML; most are non-personalized content-based — our personalized K-Means clustering differentiates us from the majority",
            "Contextual modeling (74% of CARS) outperforms pre/post-filtering → we embed distance_km directly into persona-weighted scoring, not as a post-filter",
            "Semantic clustering on multi-dimensional normalized vectors (92.8% Top-5 precision) validates K-Means on [price, rating, distance, count] over simple distance-sort",
        ],
    }
    st.dataframe(pd.DataFrame(lit_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Market Survey — Competitive Analysis")
    market_data = {
        "Platform": ["Google Maps", "Wolt", "TripAdvisor", "⭐ Appetite Engineering"],
        "Exact NIS Price": ["No ($$$ ordinal)", "Yes — delivery menu", "No ($$$ ordinal)", "Yes — per portion"],
        "Real-time GPS": ["Yes", "Delivery radius only", "No", "Yes"],
        "ML Personalization": ["None", "Category-based", "Login history", "Clustering + personas"],
        "Walk-in Optimized": ["Partial", "No", "No", "Yes"],
        "Optimizes For": ["Ad engagement", "Delivery commission", "Review traffic", "User value"],
    }
    st.dataframe(pd.DataFrame(market_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Positioning Diagram — Price Transparency vs. Personalization")

    fig_pos = go.Figure()

    # Sweet-spot quadrant
    fig_pos.add_shape(type="rect", x0=5, y0=5, x1=10.2, y1=10.2,
                      fillcolor="rgba(42,157,143,0.07)", line_width=0)
    fig_pos.add_annotation(x=7.5, y=9.7, text="Sweet spot", showarrow=False,
                           font=dict(color="#2a9d8f", size=11))

    fig_pos.add_hline(y=5, line_dash="dot", line_color="#ced4da", line_width=1)
    fig_pos.add_vline(x=5, line_dash="dot", line_color="#ced4da", line_width=1)

    competitors = [
        ("Google Maps",         1.5, 2.0,  18, "#6c757d"),
        ("Wolt",                3.5, 8.5,  18, "#6c757d"),
        ("TripAdvisor",         2.5, 1.5,  18, "#6c757d"),
        ("Appetite Engineering", 9.0, 9.2, 28, "#e76f51"),
    ]
    for name, px_val, py_val, sz, clr in competitors:
        fig_pos.add_trace(go.Scatter(
            x=[px_val], y=[py_val], mode="markers+text",
            marker=dict(size=sz, color=clr, opacity=0.9),
            text=[name], textposition="top center",
            showlegend=False,
        ))

    fig_pos.update_layout(
        xaxis_title="Personalization Level →",
        yaxis_title="Price Transparency →",
        xaxis=dict(range=[0, 10.5], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 10.5], showgrid=False, zeroline=False),
        height=460,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_pos, use_container_width=True)
    st.caption(
        "**Our differentiation:** only platform combining exact NIS prices + ML persona clustering "
        "+ walk-in real-time geolocation — occupying the uncontested top-right quadrant."
    )


# ══════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("EDA — Exploratory Data Analysis")
    st.caption(f"Sidebar filters active · showing **{len(df_rated):,}** rated venues")

    # ── Data quality checklist ─────────────────────────────────
    st.markdown("#### Data Quality Checklist")
    ck1, ck2, ck3, ck4, ck5 = st.columns(5)
    ck1.metric("Raw rows", f"{len(df_all):,}")
    ck2.metric("After clean/dedup", f"{len(df_rated):,}")
    ck3.metric("Null ratings", f"{df_all['rating'].isna().sum()}")
    ck4.metric("Cities", f"{df_all['city'].nunique()}")
    ck5.metric("Price source: estimated", f"{(df_all['price_source'] == 'estimated').sum():,}")

    with st.expander("Full schema & null counts"):
        null_df = pd.DataFrame({
            "Column": df_all.columns,
            "Dtype": df_all.dtypes.astype(str).values,
            "Non-null": df_all.notna().sum().values,
            "Nulls": df_all.isna().sum().values,
            "Null %": (df_all.isna().sum() / len(df_all) * 100).round(1).astype(str) + "%",
        })
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 3 Insights ────────────────────────────────────────────
    st.markdown("#### 3 Key Insights")

    base = df_all.dropna(subset=["rating", price_col])
    corr = base[price_col].corr(base["rating"])
    price_iqr = base[price_col].quantile(0.75) - base[price_col].quantile(0.25)
    city_means = base.groupby("city")[price_col].mean()
    city_range = city_means.max() - city_means.min()
    pct_iqr = ((base[price_col] >= base[price_col].quantile(0.25)) &
               (base[price_col] <= base[price_col].quantile(0.75))).mean() * 100

    ins1, ins2, ins3 = st.columns(3)
    ins1.info(
        f"**No price-quality correlation**\n\n"
        f"Pearson r = **{corr:.2f}** between price and rating. "
        f"A 58 NIS shawarma is statistically no better than a 44 NIS one — "
        f"the market is informationally chaotic."
    )
    ins2.info(
        f"**Prices cluster in a narrow band**\n\n"
        f"IQR = **{price_iqr:.0f} NIS** · {pct_iqr:.0f}% of venues fall within it. "
        f"Price alone is a poor differentiator; rating + distance must be weighted in."
    )
    ins3.info(
        f"**Geographic arbitrage exists**\n\n"
        f"Average price varies by **{city_range:.0f} NIS** across cities. "
        f"City context is a meaningful signal — the model needs coordinates, not just price."
    )

    st.divider()

    # ── Visualization 1 — Price distribution ──────────────────
    st.markdown("#### Visualization 1 — Price Distribution")
    fig_hist = px.histogram(
        df_rated, x=price_col, nbins=30,
        color_discrete_sequence=["#f4a261"],
        labels={price_col: "Price (NIS)", "count": "Venues"},
    )
    fig_hist.update_layout(showlegend=False, bargap=0.05, yaxis_title="Venues")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Visualization 2 & 3 ───────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Visualization 2 — Price vs. Rating (The Chaos)")
        scatter_df = df_rated[[price_col, "rating", "city", "name"]].dropna()
        coeffs = np.polyfit(scatter_df[price_col], scatter_df["rating"], 1)
        x_line = np.linspace(scatter_df[price_col].min(), scatter_df[price_col].max(), 100)
        y_line = np.polyval(coeffs, x_line)

        fig_sc = px.scatter(
            scatter_df, x=price_col, y="rating",
            color="city", hover_name="name",
            opacity=0.35,
            labels={price_col: "Price (NIS)", "rating": "Rating"},
            height=420,
        )
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="#e63946", width=2, dash="dash"),
            name=f"Trend (r={corr:.2f})",
            showlegend=True,
        ))
        fig_sc.update_traces(marker_size=4, selector=dict(mode="markers"))
        fig_sc.update_layout(showlegend=True, legend_title="")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        st.markdown("#### Visualization 3 — Avg Rating by City (Top 20)")
        city_rating = (
            df_rated.groupby("city")["rating"].agg(["mean", "count"])
            .query("count >= 5")
            .sort_values("mean", ascending=False)
            .head(20).reset_index()
            .rename(columns={"mean": "avg_rating", "count": "venues"})
        )
        fig_city = px.bar(
            city_rating, x="avg_rating", y="city", orientation="h",
            color="avg_rating", color_continuous_scale="RdYlGn", range_color=[3.5, 5.0],
            hover_data={"venues": True},
            labels={"avg_rating": "Avg Rating", "city": ""},
            height=420,
        )
        fig_city.update_layout(
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_city, use_container_width=True)

    # ── Map ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Venue Map — Rating × Price")
    df_map = df_rated.dropna(subset=["lat", "lng", "rating"])
    if not df_map.empty:
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lng",
            color="rating", color_continuous_scale="RdYlGn", range_color=[1, 5],
            size=price_col, size_max=10,
            hover_name="name",
            hover_data={"city": True, price_col: True, "rating": True,
                        "lat": False, "lng": False},
            zoom=7, center={"lat": 31.5, "lon": 34.9},
            height=520, mapbox_style="open-street-map", opacity=0.75,
        )
        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Rating", thickness=12),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No venues match the current filters.")

    # ── Top venues table ──────────────────────────────────────
    st.divider()
    st.markdown("#### Top 50 Rated Venues")
    top = (
        df_rated.sort_values("rating", ascending=False)
        .head(50)[["name", "city", "rating", price_col, "car_park_nearby", "google_maps_url"]]
        .rename(columns={
            "name": "Name", "city": "City", "rating": "Rating",
            price_col: "Price (NIS)", "car_park_nearby": "Parking",
            "google_maps_url": "Google Maps",
        })
        .reset_index(drop=True)
    )
    top.index += 1
    st.dataframe(
        top, use_container_width=True,
        column_config={
            "Google Maps": st.column_config.LinkColumn("Google Maps", display_text="Open ↗"),
            "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
            "Price (NIS)": st.column_config.NumberColumn(format="₪%d"),
        },
    )


# ══════════════════════════════════════════════════════════════
# TAB 4 — KPI & Model
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("KPI Definition")
    st.success(
        "**The model is K-Means clustering (k=5), the metric is Silhouette Score ≥ 0.45, "
        "because unsupervised venue segmentation has no ground-truth labels — "
        "cohesion/separation ratio is the only objective measure of whether clusters "
        "represent meaningfully distinct value profiles.**"
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Target Silhouette Score", "≥ 0.45", "vs. random baseline ≈ 0.10")
    k2.metric("Persona Match Rate", "≥ 90%", "manual eval · 50 test cases")
    k3.metric("Response Time", "< 3 sec", "GPS lock → ranked list")

    st.divider()
    st.subheader("Formal ML Problem Statement")
    st.markdown("""
| Component | Definition |
|-----------|-----------|
| **Input X** | `[price_NIS, rating, distance_km, ratings_count]` — StandardScaler-normalized |
| **Output y** | Cluster label per venue + persona-weighted ranking score |
| **Algorithm** | K-Means · k tuned via Elbow + Silhouette on k ∈ {3…8} |
| **Loss / Objective** | Minimize intra-cluster variance; maximize inter-cluster separation |
| **Train / Val / Test** | 70% / 15% / 15% stratified by city |
| **Baseline** | Naïve sort by distance only — current Google Maps default |
    """)

    st.divider()
    st.subheader("Why Silhouette ≥ 0.45?")
    silhouette_ref = pd.DataFrame({
        "Score range": ["< 0.25", "0.25 – 0.45", "≥ 0.45 (our target)", "≥ 0.70"],
        "Interpretation": [
            "Clusters overlap — no better than random",
            "Weak structure — personas would be unreliable",
            "Clear venue-profile separation — personas are actionable",
            "Strong structure — price signal dominates",
        ],
    })
    st.dataframe(silhouette_ref, use_container_width=True, hide_index=True)
    st.caption(
        "Precedent: Asghar (2016) reports restaurant dataset clusters achieve Silhouette 0.42–0.61 "
        "depending on feature set. Our 0.45 floor aligns with the lower bound of meaningful segmentation."
    )

    st.divider()
    st.subheader("Risk Register")
    risk_data = {
        "Risk": [
            "~35% of venues lack exact NIS prices (only Google ordinal price_level)",
            "Optimal k unclear for dense urban venue dataset",
            "Google Maps API quota overrun during scraping phase",
        ],
        "Likelihood": ["High", "Medium", "Medium"],
        "Impact": ["High", "Medium", "Low"],
        "Mitigation": [
            "Impute from price_level median per tier; flag imputed venues in UI with a warning badge",
            "Sweep k ∈ {3…8} with Elbow + Silhouette on validation set; fall back to DBSCAN if K-Means fails",
            "Cache all responses in data/raw/; run off-peak within $200 free-tier credit; throttle to 1 req/s",
        ],
    }
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    model, scaler = get_model()
    st.divider()
    st.subheader("Recommendation Model Status")
    st.markdown(
        "This app uses a trained K-Means model on venue attributes plus location distance. "
        "If no serialized model is available, it trains one automatically from the current dataset."
    )
    st.markdown(f"**Model artifacts:** `{MODEL_PATH.name}` / `{SCALER_PATH.name}`")
    st.info("Run `python -m src.train` from the repo root to regenerate the saved model and evaluation report.")

with tab5:
    st.subheader("Personalized Shawarma Recommendations")
    st.markdown(
        "Enter your current GPS coordinates and choose a persona. The ranked list uses the trained model "
        "alongside a persona-weighted score that balances price, rating, and walking distance."
    )
    col_a, col_b = st.columns(2)

    with col_a:
        persona = st.selectbox(
            "Persona",
            ["student", "quality"],
            format_func=lambda label: "Thrifty Student" if label == "student" else "Quality Enthusiast",
        )
        user_lat = st.number_input("Current latitude", value=31.5, format="%.6f")
        user_lng = st.number_input("Current longitude", value=34.9, format="%.6f")
        max_dist_km = st.slider("Maximum walking distance (km)", 0.5, 5.0, 2.0, step=0.1)

    with col_b:
        product_label = st.selectbox("Reference product", list(PRICE_LABELS.keys()))
        price_col = PRICE_LABELS[product_label]
        st.markdown(
            "The model uses the selected NIS price column as the price feature for ranking. "
            "If a venue is missing that price, it is excluded from the recommendation set."
        )

    if st.button("Recommend"):
        df_user = build_features(df_all, user_lat, user_lng)
        df_user["price_nis"] = df_user[price_col]
        df_user["ratings_count"] = df_user["reviews_count"].fillna(0).astype(int)
        df_user = df_user.dropna(subset=["price_nis", "rating"])

        df_recommend = predict(model, scaler, df_user, persona=persona, max_dist_km=max_dist_km)
        if df_recommend.empty:
            st.warning("No venues were found within the selected distance and feature filters.")
        else:
            baseline = df_user.sort_values("distance_km").head(10).copy()
            weights = {
                "student": {"price_nis": -1.5, "rating": 1.0, "distance_km": -0.8},
                "quality": {"price_nis": -0.5, "rating": 2.0, "distance_km": -0.5},
            }[persona]
            baseline["score"] = (
                weights["rating"] * baseline["rating"]
                + weights["price_nis"] * baseline["price_nis"] / 10
                + weights["distance_km"] * baseline["distance_km"]
            )

            st.metric("Top recommendation", df_recommend.iloc[0]["name"])
            st.metric("Top 10 average model score", f"{df_recommend.head(10)['score'].mean():.2f}")
            st.metric("Top 10 average distance-only score", f"{baseline['score'].mean():.2f}")

            result_cols = ["name", "city", "rating", "price_nis", "distance_km", "cluster", "score", "google_maps_url"]
            output = df_recommend.head(20)[result_cols].rename(
                columns={
                    "name": "Name",
                    "city": "City",
                    "rating": "Rating",
                    "price_nis": "Price (NIS)",
                    "distance_km": "Distance (km)",
                    "cluster": "Cluster",
                    "score": "Persona Score",
                    "google_maps_url": "Google Maps",
                }
            )
            output.index += 1
            st.dataframe(
                output,
                use_container_width=True,
                column_config={
                    "Google Maps": st.column_config.LinkColumn("Google Maps", display_text="Open ↗"),
                    "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
                    "Price (NIS)": st.column_config.NumberColumn(format="₪%d"),
                    "Distance (km)": st.column_config.NumberColumn(format="%.2f"),
                    "Persona Score": st.column_config.NumberColumn(format="%.2f"),
                },
            )
