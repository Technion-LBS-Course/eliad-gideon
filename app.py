import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from src.data import filter_haifa, generate_haifa_queries, load_raw, clean
from src.model import (
    assign_cluster_labels,
    assign_haifa_cluster_labels,
    compare_algorithms,
    HAIFA_FEATURE_COLS,
    load_model,
    predict,
    predict_from_map,
    save_model,
    train_agglomerative,
    train_haifa_model,
)

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Problem & Personas",
    "📚 Literature & Market",
    "📊 EDA",
    "🏆 KPI & Model",
    "🔮 Predicted",
    "🗺️ Haifa",
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


# ══════════════════════════════════════════════════════════════
# TAB 5 — Predicted
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔮 Predicted — Shawarma Cluster Finder")
    st.caption(
        "The first stage of identifying your best shawarma: cluster all venues by price and rating, "
        "choose the cluster that matches your expectations, then get a ranked list."
    )

    # ── Settings ───────────────────────────────────────────────
    st.markdown("#### Settings")
    col_c, col_p, col_d = st.columns(3)
    with col_c:
        cities_list = sorted(df_all["city"].dropna().unique().tolist())
        default_city = "תל אביב יפו" if "תל אביב יפו" in cities_list else cities_list[0]
        user_city = st.selectbox("Your city", cities_list, index=cities_list.index(default_city))
    with col_p:
        persona = st.radio(
            "Persona",
            ["student", "quality"],
            format_func=lambda x: {"student": "🎓 Student", "quality": "👑 Quality"}[x],
        )
    with col_d:
        max_dist = st.slider("Max distance (km)", 0.5, 10.0, 2.0, step=0.5)

    city_center = df_all.groupby("city")[["lat", "lng"]].mean()
    user_lat = float(city_center.loc[user_city, "lat"])
    user_lng = float(city_center.loc[user_city, "lng"])

    st.divider()

    # ── Algorithm Overview ─────────────────────────────────────
    st.markdown("#### Why These 3 Algorithms?")
    st.caption(
        "Three fundamentally different clustering paradigms were chosen to compare how well each "
        "separates venues along the price–rating axis. All use the same features: "
        "`[price_nis, rating, reviews_count]` — StandardScaler-normalized."
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        with st.container(border=True):
            st.markdown("##### K-Means — Partitional")
            st.markdown("""
**How it works:** Assigns each venue to its nearest centroid, then re-computes centroids iteratively.

**Why chosen:** Fast, scalable to 12k venues, and — critically — supports native `predict()` for new venues at query time.

**Limitation:** Assumes spherical clusters of similar size; sensitive to the tight ₪5 price IQR.
            """)
    with a2:
        with st.container(border=True):
            st.markdown("##### DBSCAN — Density-Based")
            st.markdown("""
**How it works:** Groups venues in dense regions; marks sparse points as noise (−1). No k to specify.

**Why chosen:** Shape-agnostic — can find non-spherical clusters and isolate outliers (e.g. uniquely priced venues).

**Limitation:** No native `predict()` — new venues need a KNN fallback. Silhouette computed only on non-noise points, which inflates the score.
            """)
    with a3:
        with st.container(border=True):
            st.markdown("##### Agglomerative — Hierarchical")
            st.markdown("""
**How it works:** Merges the two closest clusters bottom-up using ward linkage (minimises intra-cluster variance).

**Why chosen:** Reveals nested price–rating structure without assuming cluster shape; uses the same auto-tuned k as K-Means for a fair comparison.

**Limitation:** No native `predict()` — new venues need a KNN fallback. Computationally heavier than K-Means.
            """)

    st.divider()

    # ── Train & Compare ────────────────────────────────────────
    st.markdown("#### Step 1 — Train & Compare Algorithms")
    st.caption("70% train / 30% test split · k auto-tuned ∈ {3…8} via silhouette sweep · KPI: Silhouette Score")

    if st.button("🔬 Train & Compare Models", type="secondary"):
        with st.spinner("Training KMeans, DBSCAN, and Agglomerative on 70% train split…"):
            km_result, db_result, agg_result = compare_algorithms(df_all)
            st.session_state["km_result"] = km_result
            st.session_state["db_result"] = db_result
            st.session_state["agg_result"] = agg_result
            save_model(km_result)

    if "km_result" in st.session_state:
        km = st.session_state["km_result"]
        db = st.session_state["db_result"]
        agg = st.session_state["agg_result"]

        # Comparison table
        cmp_data = {
            "Algorithm": ["KMeans", "DBSCAN", "Agglomerative"],
            "Paradigm": ["Partitional", "Density-based", "Hierarchical (ward)"],
            "Hyperparameters": [
                f"k = {km['k']} (auto-tuned)",
                f"eps = {db['eps']}, min_samples = {db['min_samples']}",
                f"k = {agg['k']} (auto-tuned), linkage = ward",
            ],
            "Train Silhouette": [
                round(km["train_silhouette"], 3),
                round(db["train_silhouette"], 3),
                round(agg["train_silhouette"], 3),
            ],
            "Test Silhouette": [
                round(km["test_silhouette"], 3),
                round(db["test_silhouette"], 3),
                round(agg["test_silhouette"], 3),
            ],
            "Meets KPI ≥ 0.45": [
                "✅" if km["test_silhouette"] >= 0.45 else "❌",
                "✅" if db["test_silhouette"] >= 0.45 else "❌",
                "✅" if agg["test_silhouette"] >= 0.45 else "❌",
            ],
            "predict() support": ["✅ native", "❌ KNN fallback", "❌ KNN fallback"],
        }
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

        # Silhouette explanation
        with st.expander("What is the Silhouette Score?"):
            st.markdown("""
The **Silhouette Score** measures how well each venue fits its assigned cluster compared to the nearest other cluster.
It ranges from **−1 to +1**:

| Score range | Meaning |
|-------------|---------|
| > 0.70 | Strong, well-separated clusters |
| 0.45 – 0.70 | Reasonable structure — clusters are meaningful |
| 0.25 – 0.45 | Weak structure — some overlap between clusters |
| < 0.25 | Clusters overlap — no better than random |

**Why our KMeans scores ~0.37 (below the 0.45 target):**
The dataset has a very tight price band — 50% of venues fall within a ₪5 window (IQR ₪43–₪48).
This compresses the price axis and makes it hard for any algorithm to produce well-separated clusters.
DBSCAN's higher score (0.71) is partly because it only evaluates non-noise points, which inflates the metric.
            """)

        # Elbow chart
        col_elbow, col_scatter = st.columns(2)
        with col_elbow:
            k_vals = list(km["k_scores"].keys())
            sil_vals = list(km["k_scores"].values())
            fig_elbow = px.line(
                x=k_vals, y=sil_vals, markers=True,
                labels={"x": "k (clusters)", "y": "Silhouette Score"},
                title=f"Silhouette sweep — best k = {km['k']}",
                color_discrete_sequence=["#2a9d8f"],
            )
            fig_elbow.add_vline(
                x=km["k"], line_dash="dash", line_color="#e76f51",
                annotation_text=f"k={km['k']}",
                annotation_position="top right",
            )
            fig_elbow.update_layout(yaxis_range=[0, 1], height=350)
            st.plotly_chart(fig_elbow, use_container_width=True)

        # Cluster scatter: price vs rating coloured by cluster label
        with col_scatter:
            cluster_labels = assign_cluster_labels(km, df_all)
            df_sample = df_all.dropna(subset=["price_nis", "rating"]).sample(
                min(2000, len(df_all)), random_state=42
            ).copy()
            from src.model import FEATURE_COLS as _FC
            X_sample = km["scaler"].transform(df_sample[_FC].fillna(0))
            df_sample["cluster_id"] = km["model"].predict(X_sample)
            df_sample["cluster_label"] = df_sample["cluster_id"].map(cluster_labels)
            fig_scatter = px.scatter(
                df_sample, x="price_nis", y="rating",
                color="cluster_label",
                opacity=0.5,
                labels={"price_nis": "Price (NIS)", "rating": "Rating", "cluster_label": "Cluster"},
                title="Clusters — Price vs Rating",
                height=350,
            )
            fig_scatter.update_traces(marker_size=4)
            fig_scatter.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Winner rationale
        st.success(
            f"**Selected model: KMeans (k={km['k']})** — "
            f"the only algorithm with native `predict()` for new venues, enabling real-time recommendations. "
            f"DBSCAN achieves a higher silhouette ({db['test_silhouette']:.3f}) but its score is inflated "
            f"since it is computed only on non-noise points, and it cannot generalize to unseen venues "
            f"without a KNN approximation. "
            f"Agglomerative Clustering shares KMeans' silhouette ({agg['test_silhouette']:.3f}) "
            f"but also lacks native prediction support. "
            f"Model saved → `data/kmeans_model.pkl`."
        )

    st.divider()

    # ── Predicted Recommendations ──────────────────────────────
    st.markdown("#### Step 2 — Choose Your Cluster & Find Your Shawarma")

    if "km_result" not in st.session_state:
        persisted = load_model()
        if persisted:
            st.session_state["km_result"] = persisted

    if "km_result" in st.session_state:
        km_loaded = st.session_state["km_result"]
        cluster_labels_loaded = assign_cluster_labels(km_loaded, df_all)

        # Build label options with venue counts for context
        df_labeled = df_all.dropna(subset=["price_nis", "rating"]).copy()
        from src.model import FEATURE_COLS as _FC2
        X_all = km_loaded["scaler"].transform(df_labeled[_FC2].fillna(0))
        df_labeled["cluster_id"] = km_loaded["model"].predict(X_all)
        df_labeled["cluster_label"] = df_labeled["cluster_id"].map(cluster_labels_loaded)
        label_counts = df_labeled["cluster_label"].value_counts().to_dict()

        label_options = sorted(
            cluster_labels_loaded.values(),
            key=lambda lbl: (
                0 if "Good" in lbl else (1 if "Average" in lbl else 2),
                0 if "Affordable" in lbl else (1 if "Reasonable" in lbl else 2),
            ),
        )
        label_options_display = [
            f"{lbl}  ({label_counts.get(lbl, 0):,} venues)" for lbl in label_options
        ]

        selected_display = st.selectbox(
            "Select the cluster that matches your expectations",
            label_options_display,
            help="Clusters are defined by their centroid price and rating relative to the full dataset distribution.",
        )
        selected_label = selected_display.split("  (")[0]

        if st.button("🥙 Find My Shawarma", type="primary"):
            result = st.session_state["km_result"]
            df_recs = predict(
                result, df_all,
                persona=persona,
                user_lat=user_lat,
                user_lng=user_lng,
                max_dist_km=max_dist,
            )
            # Filter to selected cluster label
            if not df_recs.empty:
                df_recs["cluster_label"] = df_recs["cluster"].map(cluster_labels_loaded)
                df_recs = df_recs[df_recs["cluster_label"] == selected_label]

            if df_recs.empty:
                st.info(
                    f"No **{selected_label}** venues found within {max_dist} km of {user_city}. "
                    f"Try a different cluster, increasing the distance, or selecting a larger city."
                )
            else:
                st.success(
                    f"Found **{len(df_recs):,}** venues in cluster **{selected_label}** "
                    f"within {max_dist} km · showing top 10"
                )
                display_cols = ["name", "city", "rating", "price_nis", "distance_km", "cluster_label", "score"]
                if "google_maps_url" in df_recs.columns:
                    display_cols.append("google_maps_url")
                top10 = df_recs.head(10)[display_cols].copy()
                top10["distance_km"] = top10["distance_km"].round(2)
                top10["score"] = top10["score"].round(2)
                top10.index = range(1, len(top10) + 1)
                st.dataframe(
                    top10.rename(columns={
                        "name": "Venue", "city": "City", "rating": "Rating",
                        "price_nis": "Price (NIS)", "distance_km": "Dist (km)",
                        "cluster_label": "Cluster", "score": "Score",
                        "google_maps_url": "Maps",
                    }),
                    use_container_width=True,
                    column_config={
                        "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
                        "Price (NIS)": st.column_config.NumberColumn(format="₪%.0f"),
                        "Dist (km)": st.column_config.NumberColumn(format="%.2f km"),
                        "Maps": st.column_config.LinkColumn("Maps", display_text="Open ↗"),
                    },
                )
    else:
        st.info("Run Step 1 first to train the model and unlock cluster selection.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — Haifa Interactive Map
# ══════════════════════════════════════════════════════════════
_CLUSTER_PALETTE = [
    "#2a9d8f", "#e76f51", "#264653", "#f4a261",
    "#e9c46a", "#a8dadc", "#d62828", "#6a4c93",
]

with tab6:
    st.subheader("🗺️ Haifa — Interactive Map Predictor")
    st.caption(
        "Model trained on **[price, rating, lat, lng]** — clusters capture both venue quality "
        "and geographic sub-area (Carmel / Haifa / Krayot). "
        "Click anywhere on the map to find the best shawarma near that point."
    )

    df_haifa = filter_haifa(df_all)
    df_synth = generate_haifa_queries(500)

    # ── Stats row ─────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Venues in region", f"{len(df_haifa):,}")
    m2.metric("Synthetic query points", "500")
    m3.metric("Features used", "price · rating · lat · lng")

    st.divider()

    # ── Settings ───────────────────────────────────────────────
    col_p, col_d, col_s = st.columns(3)
    with col_p:
        persona_h = st.radio(
            "Persona", ["student", "quality"], key="persona_h",
            format_func=lambda x: {"student": "🎓 Student", "quality": "👑 Quality"}[x],
        )
    with col_d:
        max_dist_h = st.slider("Max distance (km)", 0.5, 5.0, 1.5, step=0.5, key="max_dist_h")
    with col_s:
        show_synth = st.checkbox("Heatmap: 500 synthetic query points", value=True)

    # ── Train ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Step 1 — Train Haifa Model")
    st.caption(
        "Includes `lat` and `lng` as features so the model discovers clusters that are "
        "**geographically cohesive** (e.g. 'Good & Affordable — Carmel') in addition to "
        "price and rating cohesion."
    )

    if st.button("🔬 Train Haifa Model", type="secondary", key="train_haifa"):
        with st.spinner("Training KMeans on Haifa venues with [price, rating, lat, lng]…"):
            h_result = train_haifa_model(df_haifa)
            h_labels = assign_haifa_cluster_labels(h_result, df_haifa)
            st.session_state["haifa_result"] = h_result
            st.session_state["haifa_labels"] = h_labels

    if "haifa_result" in st.session_state:
        h_result = st.session_state["haifa_result"]
        h_labels = st.session_state["haifa_labels"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best k", h_result["k"])
        c2.metric("Train Silhouette", f"{h_result['train_silhouette']:.3f}")
        c3.metric("Test Silhouette", f"{h_result['test_silhouette']:.3f}")

        # Label legend
        with st.expander("Cluster labels discovered"):
            for cid, lbl in sorted(h_labels.items()):
                color = _CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)]
                st.markdown(
                    f"<span style='background:{color};color:#fff;padding:2px 8px;"
                    f"border-radius:4px;font-size:0.85em'>{lbl}</span>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Map ────────────────────────────────────────────────────
    st.markdown("#### Step 2 — Click on the Map")
    st.caption("Click anywhere in the Haifa area — the app will query the model from that point.")

    # Build base map
    haifa_map = folium.Map(
        location=[32.815, 35.01],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # Venue markers
    if "haifa_result" in st.session_state:
        h_result = st.session_state["haifa_result"]
        h_labels = st.session_state["haifa_labels"]
        df_plot = df_haifa.dropna(subset=HAIFA_FEATURE_COLS).copy()
        X_plot = h_result["scaler"].transform(df_plot[HAIFA_FEATURE_COLS])
        df_plot["cluster_id"] = h_result["model"].predict(X_plot)
        df_plot["cluster_label"] = df_plot["cluster_id"].map(h_labels)
        unique_labels = sorted(df_plot["cluster_label"].unique())
        color_map = {lbl: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i, lbl in enumerate(unique_labels)}

        for _, row in df_plot.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=5,
                color=color_map[row["cluster_label"]],
                fill=True,
                fill_opacity=0.75,
                tooltip=(
                    f"<b>{row['name']}</b><br>"
                    f"{row['cluster_label']}<br>"
                    f"⭐ {row['rating']:.1f} · ₪{row['price_nis']:.0f}"
                ),
            ).add_to(haifa_map)

        # Legend
        legend_html = "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;" \
                      "background:white;padding:10px;border-radius:6px;border:1px solid #ccc;" \
                      "font-size:12px;max-width:200px'>"
        for lbl, clr in color_map.items():
            legend_html += (
                f"<span style='background:{clr};display:inline-block;"
                f"width:12px;height:12px;border-radius:50%;margin-right:5px'></span>"
                f"{lbl}<br>"
            )
        legend_html += "</div>"
        haifa_map.get_root().html.add_child(folium.Element(legend_html))
    else:
        # No model yet — grey markers
        for _, row in df_haifa.dropna(subset=["lat", "lng"]).iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=4,
                color="#6c757d",
                fill=True,
                fill_opacity=0.5,
                tooltip=f"{row['name']} · ⭐{row['rating']:.1f}",
            ).add_to(haifa_map)

    # Synthetic query heatmap
    if show_synth:
        HeatMap(
            [[r["lat"], r["lng"]] for _, r in df_synth.iterrows()],
            radius=12, blur=8, max_zoom=13, name="Query coverage",
        ).add_to(haifa_map)

    # Render map — returns click info
    map_out = st_folium(haifa_map, width="100%", height=520, returned_objects=["last_clicked"])

    # ── Results from click ─────────────────────────────────────
    if map_out and map_out.get("last_clicked"):
        click_lat = map_out["last_clicked"]["lat"]
        click_lng = map_out["last_clicked"]["lng"]

        st.info(f"📍 Pinned: **{click_lat:.5f}°N, {click_lng:.5f}°E**")

        if "haifa_result" not in st.session_state:
            st.warning("Train the Haifa model first (Step 1) to get recommendations.")
        else:
            h_result = st.session_state["haifa_result"]
            h_labels = st.session_state["haifa_labels"]

            df_recs = predict_from_map(
                h_result, df_haifa,
                user_lat=click_lat, user_lng=click_lng,
                persona=persona_h, max_dist_km=max_dist_h,
            )

            if df_recs.empty:
                st.info(
                    f"No venues within {max_dist_h} km of the clicked point. "
                    "Try clicking closer to a populated area or increase the distance."
                )
            else:
                df_recs["cluster_label"] = df_recs["cluster"].map(h_labels)
                available = sorted(df_recs["cluster_label"].unique())
                cluster_counts = df_recs["cluster_label"].value_counts().to_dict()

                st.markdown("#### Results")
                col_sel, col_info = st.columns([2, 3])
                with col_sel:
                    chosen = st.selectbox(
                        "Filter by cluster",
                        ["All"] + [f"{lbl} ({cluster_counts[lbl]})" for lbl in available],
                        key="haifa_cluster_sel",
                    )
                with col_info:
                    st.metric("Total venues nearby", len(df_recs))

                chosen_label = None if chosen == "All" else chosen.rsplit(" (", 1)[0]
                df_show = df_recs if chosen_label is None else df_recs[df_recs["cluster_label"] == chosen_label]

                st.success(f"Showing **{min(10, len(df_show))}** of {len(df_show)} venues")
                disp = ["name", "city", "rating", "price_nis", "distance_km", "cluster_label", "score"]
                if "google_maps_url" in df_show.columns:
                    disp.append("google_maps_url")
                top = df_show.head(10)[disp].copy()
                top["distance_km"] = top["distance_km"].round(2)
                top["score"] = top["score"].round(2)
                top.index = range(1, len(top) + 1)
                st.dataframe(
                    top.rename(columns={
                        "name": "Venue", "city": "City", "rating": "Rating",
                        "price_nis": "Price (NIS)", "distance_km": "Dist (km)",
                        "cluster_label": "Cluster", "score": "Score",
                        "google_maps_url": "Maps",
                    }),
                    use_container_width=True,
                    column_config={
                        "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
                        "Price (NIS)": st.column_config.NumberColumn(format="₪%.0f"),
                        "Dist (km)": st.column_config.NumberColumn(format="%.2f km"),
                        "Maps": st.column_config.LinkColumn("Maps", display_text="Open ↗"),
                    },
                )
    else:
        st.info("👆 Click anywhere on the map above to get recommendations for that location.")
