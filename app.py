import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.agent import extract_params, fallback_recommend, recommend
from src.data import load_raw, clean
from src.model import (
    assign_cluster_labels,
    compare_algorithms,
    compute_confusion_matrix,
    load_model,
    save_model,
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
    "🤖 Agent",
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
        "**The model is K-Means clustering (k=9, fixed — one cluster per target class), "
        "evaluated against a 9-class target (score tier × price tier). "
        "Primary metric: Silhouette Score (unsupervised cohesion/separation). "
        "Secondary metric: Confusion Matrix accuracy on the held-out 20% test set.**"
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Target Silhouette Score", "≥ 0.45", "vs. random baseline — computed per run")
    k2.metric("Persona Match Rate", "≥ 90%", "manual eval · 50 test cases")
    k3.metric("Response Time", "< 3 sec", "GPS lock → ranked list")

    st.divider()
    st.subheader("Formal ML Problem Statement")
    st.markdown("""
| Component | Definition |
|-----------|-----------|
| **Input X** | `[price_NIS, weighted_rating]` — Bayesian-smoothed rating + price, StandardScaler-normalized |
| **Output y** | 9-class label: {good / medium / bad score} × {high / fair / low price} |
| **Algorithm** | K-Means · k = 9 (fixed — one cluster per target class) |
| **Loss / Objective** | Minimize intra-cluster variance; maximize inter-cluster separation |
| **Train / Test** | 80% / 20% random split (`random_state=42`) |
| **Baseline** | Random cluster assignment (same k = 9) — computed per run |
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
    st.subheader("🔮 Predicted — Shawarma Cluster Analysis")
    st.caption(
        "Cluster all venues by **confidence-weighted rating** and **price**, then explore what features "
        "define each cluster. Rating is Bayesian-smoothed by review count: venues with fewer reviews "
        "are pulled toward the global mean — high review count = high confidence."
    )

    st.divider()

    # ── Algorithm Overview ─────────────────────────────────────
    st.markdown("#### Why These 3 Algorithms?")
    st.caption(
        "Three fundamentally different clustering paradigms were chosen to compare how well each "
        "separates venues along the price–weighted_rating axis. "
        "Features: `[price_nis, weighted_rating]` — StandardScaler-normalized."
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
    st.caption("80% train / 20% test split · k = 9 (fixed — one cluster per target class) · KPI: Silhouette Score + Confusion Matrix accuracy")

    if st.button("🔬 Train & Compare Models", type="secondary"):
        with st.spinner("Training KMeans, DBSCAN, and Agglomerative on 80% train split…"):
            km_result, db_result, agg_result, df_tr, df_te = compare_algorithms(df_all)
            st.session_state["km_result"] = km_result
            st.session_state["db_result"] = db_result
            st.session_state["agg_result"] = agg_result
            st.session_state["df_train"] = df_tr
            st.session_state["df_test"] = df_te
            save_model(km_result)

    if "km_result" in st.session_state and "db_result" in st.session_state and "agg_result" in st.session_state:
        km = st.session_state["km_result"]
        db = st.session_state["db_result"]
        agg = st.session_state["agg_result"]

        # Comparison table — includes random baseline row so the panel can verify we beat it
        baseline_sil = round(km.get("baseline_silhouette", 0.0), 3)
        cmp_data = {
            "Algorithm": ["Random baseline", "KMeans", "DBSCAN", "Agglomerative"],
            "Paradigm": ["—", "Partitional", "Density-based", "Hierarchical (ward)"],
            "Hyperparameters": [
                f"k = {km['k']} (same as KMeans), random labels",
                f"k = {km['k']} (fixed — 9 target classes)",
                f"eps = {db['eps']:.3f} (auto-tuned), min_samples = {db['min_samples']}",
                f"k = {agg['k']} (fixed), linkage = ward",
            ],
            "Train Silhouette": ["—", round(km["train_silhouette"], 3), round(db["train_silhouette"], 3), round(agg["train_silhouette"], 3)],
            "Test Silhouette": [baseline_sil, round(km["test_silhouette"], 3), round(db["test_silhouette"], 3), round(agg["test_silhouette"], 3)],
            "Beats baseline": [
                "—",
                "✅" if km["test_silhouette"] > baseline_sil else "❌",
                "✅" if db["test_silhouette"] > baseline_sil else "❌",
                "✅" if agg["test_silhouette"] > baseline_sil else "❌",
            ],
            "Meets KPI ≥ 0.45": [
                "❌",
                "✅" if km["test_silhouette"] >= 0.45 else "❌",
                "✅" if db["test_silhouette"] >= 0.45 else "❌",
                "✅" if agg["test_silhouette"] >= 0.45 else "❌",
            ],
            "predict() support": ["—", "✅ native", "❌ KNN fallback", "❌ KNN fallback"],
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

        # Target class vs cluster scatter (two-panel comparison)
        col_truth, col_pred = st.columns(2)
        from src.model import FEATURE_COLS as _FC

        df_sample = df_all.dropna(subset=["price_nis", "weighted_rating"]).sample(
            min(2000, len(df_all)), random_state=42
        ).copy()
        # True classes on sample
        if "df_train" in st.session_state:
            df_tr_s = st.session_state["df_train"]
            p33_s = float(df_tr_s["price_nis"].quantile(0.33))
            p67_s = float(df_tr_s["price_nis"].quantile(0.67))
        else:
            p33_s = float(df_all["price_nis"].quantile(0.33))
            p67_s = float(df_all["price_nis"].quantile(0.67))
        score_s = np.select(
            [df_sample["weighted_rating"] >= 4.5, df_sample["weighted_rating"] >= 3.0],
            ["good score", "medium score"], default="bad score",
        )
        price_s = np.select(
            [df_sample["price_nis"] >= p67_s, df_sample["price_nis"] >= p33_s],
            ["high price", "fair price"], default="low price",
        )
        df_sample["target_class"] = score_s + " - " + price_s

        with col_truth:
            fig_truth = px.scatter(
                df_sample, x="price_nis", y="weighted_rating",
                color="target_class",
                opacity=0.45,
                labels={"price_nis": "Price (NIS)", "weighted_rating": "Conf. Rating", "target_class": "True Class"},
                title="Ground truth — 9 target classes",
                height=380,
            )
            fig_truth.add_hline(y=4.5, line_dash="dot", line_color="rgba(0,0,0,0.3)", annotation_text="4.5")
            fig_truth.add_hline(y=3.0, line_dash="dot", line_color="rgba(0,0,0,0.3)", annotation_text="3.0")
            fig_truth.update_traces(marker_size=4)
            fig_truth.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.55, font_size=9))
            st.plotly_chart(fig_truth, use_container_width=True)

        cluster_labels = assign_cluster_labels(km, df_all)
        X_sample = km["scaler"].transform(df_sample[_FC].fillna(0))
        df_sample["cluster_id"] = km["model"].predict(X_sample)
        df_sample["cluster_label"] = df_sample["cluster_id"].map(cluster_labels)
        with col_pred:
            fig_pred = px.scatter(
                df_sample, x="price_nis", y="weighted_rating",
                color="cluster_label",
                opacity=0.45,
                labels={"price_nis": "Price (NIS)", "weighted_rating": "Conf. Rating", "cluster_label": "KMeans Cluster"},
                title="KMeans prediction — 9 clusters",
                height=380,
            )
            fig_pred.add_hline(y=4.5, line_dash="dot", line_color="rgba(0,0,0,0.3)", annotation_text="4.5")
            fig_pred.add_hline(y=3.0, line_dash="dot", line_color="rgba(0,0,0,0.3)", annotation_text="3.0")
            fig_pred.update_traces(marker_size=4)
            fig_pred.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.55, font_size=9))
            st.plotly_chart(fig_pred, use_container_width=True)

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

        # ── Confusion Matrix ───────────────────────────────────
        st.divider()
        st.markdown("#### Confusion Matrix — KMeans vs Target Classes (test set)")
        st.caption(
            "Cluster IDs are mapped to target classes via **Hungarian matching** on the training set "
            "(finds the 1-to-1 assignment that maximises overlap). Evaluated on the held-out 20% test set."
        )

        if "df_train" in st.session_state and "df_test" in st.session_state:
            with st.spinner("Computing confusion matrix…"):
                cm_data = compute_confusion_matrix(
                    km,
                    st.session_state["df_train"],
                    st.session_state["df_test"],
                )
            cm_arr = cm_data["confusion_matrix"]
            cm_classes = cm_data["classes"]
            acc = cm_data["accuracy"]

            st.metric("Test accuracy (class match rate)", f"{acc:.1%}")

            # Normalised heatmap
            cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
            fig_cm = px.imshow(
                cm_norm,
                x=cm_classes,
                y=cm_classes,
                color_continuous_scale="Blues",
                zmin=0, zmax=1,
                text_auto=".0%",
                aspect="auto",
                labels={"x": "Predicted (KMeans)", "y": "True class", "color": "Row %"},
                title=f"Confusion Matrix — normalised by row · accuracy {acc:.1%}",
                height=520,
            )
            fig_cm.update_xaxes(tickangle=35)
            fig_cm.update_layout(margin=dict(l=0, r=0, b=120))
            st.plotly_chart(fig_cm, use_container_width=True)

            with st.expander("Raw counts"):
                st.dataframe(
                    pd.DataFrame(cm_arr, index=cm_classes, columns=cm_classes),
                    use_container_width=True,
                )
        else:
            st.info("Confusion matrix will appear after training.")

        # ── Cluster Analysis Dashboard ─────────────────────────
        st.divider()
        st.markdown("#### Cluster Analysis — What Defines Each Cluster?")
        st.caption(
            "Rating thresholds: **Good** ≥ 4.5 · **Medium** 3.0–4.5 · **Bad** < 3.0 (confidence-adjusted). "
            "Price thresholds: **High** > 67th pct · **Fair** 33–67th pct · **Low** < 33rd pct."
        )

        from src.model import FEATURE_COLS as _FC3
        cluster_labels_full = assign_cluster_labels(km, df_all)
        df_full = df_all.dropna(subset=["price_nis", "weighted_rating"]).copy()
        X_full = km["scaler"].transform(df_full[_FC3].fillna(0))
        df_full["cluster_id"] = km["model"].predict(X_full)
        df_full["cluster_label"] = df_full["cluster_id"].map(cluster_labels_full)

        # Summary table — one row per cluster
        summary_rows = []
        for cid, lbl in sorted(cluster_labels_full.items()):
            df_c = df_full[df_full["cluster_id"] == cid]
            summary_rows.append({
                "Cluster": lbl,
                "Venues": len(df_c),
                "Avg Price (₪)": round(df_c["price_nis"].mean(), 1),
                "Price IQR (₪)": round(df_c["price_nis"].quantile(0.75) - df_c["price_nis"].quantile(0.25), 1),
                "Avg Raw Rating": round(df_c["rating"].mean(), 2),
                "Avg Conf. Rating": round(df_c["weighted_rating"].mean(), 2),
                "Avg Reviews": int(df_c["ratings_count"].mean()),
            })
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Raw Rating": st.column_config.NumberColumn(format="%.2f ⭐"),
                "Avg Conf. Rating": st.column_config.NumberColumn(format="%.2f ⭐"),
                "Avg Price (₪)": st.column_config.NumberColumn(format="₪%.1f"),
                "Price IQR (₪)": st.column_config.NumberColumn(format="₪%.1f"),
            },
        )

        # Full scatter: all venues, size = reviews_count where meaningful
        has_reviews = df_full["ratings_count"].gt(0).any()
        fig_full = px.scatter(
            df_full.sample(min(4000, len(df_full)), random_state=42),
            x="price_nis",
            y="weighted_rating",
            color="cluster_label",
            size="ratings_count" if has_reviews else None,
            size_max=15,
            opacity=0.45,
            hover_name="name",
            hover_data={"city": True, "price_nis": True, "rating": True,
                        "weighted_rating": True, "ratings_count": True,
                        "cluster_label": False},
            labels={
                "price_nis": "Price (NIS)",
                "weighted_rating": "Confidence-weighted Rating",
                "ratings_count": "Reviews",
                "cluster_label": "Cluster",
            },
            title="All venues — Price × Confidence-weighted Rating (point size = review count)",
            height=480,
        )
        fig_full.add_hline(y=4.5, line_dash="dot", line_color="#2a9d8f",
                           annotation_text="4.5 — Good threshold")
        fig_full.add_hline(y=3.0, line_dash="dot", line_color="#e76f51",
                           annotation_text="3.0 — Poor threshold")
        fig_full.update_traces(marker_line_width=0)
        fig_full.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig_full, use_container_width=True)

        # Per-cluster deep-dives
        st.markdown("#### Per-Cluster Deep-Dive")
        _RATING_EMOJI = {"Good": "🟢", "Average": "🟡", "Poor": "🔴"}
        _PRICE_EMOJI = {"Expensive": "💸", "Reasonable": "💰", "Affordable": "🪙"}

        for cid, lbl in sorted(cluster_labels_full.items()):
            df_c = df_full[df_full["cluster_id"] == cid]
            rating_tier = next((t for t in ("Good", "Average", "Poor") if t in lbl), "")
            price_tier = next((t for t in ("Expensive", "Reasonable", "Affordable") if t in lbl), "")
            emoji = _RATING_EMOJI.get(rating_tier, "⚪") + " " + _PRICE_EMOJI.get(price_tier, "")

            with st.expander(f"{emoji}  **{lbl}** — {len(df_c):,} venues", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Venues", f"{len(df_c):,}")
                c2.metric("Avg Price", f"₪{df_c['price_nis'].mean():.0f}")
                c3.metric("Price range", f"₪{df_c['price_nis'].min():.0f}–₪{df_c['price_nis'].max():.0f}")
                c4.metric("Raw Rating", f"{df_c['rating'].mean():.2f} ⭐")
                c5.metric("Conf. Rating", f"{df_c['weighted_rating'].mean():.2f} ⭐",
                          delta=f"{df_c['weighted_rating'].mean() - df_c['rating'].mean():.2f} vs raw",
                          delta_color="normal")

                st.caption(
                    f"**Price IQR:** ₪{df_c['price_nis'].quantile(0.25):.0f}–₪{df_c['price_nis'].quantile(0.75):.0f}  ·  "
                    f"**Rating std:** {df_c['rating'].std():.2f}  ·  "
                    f"**Avg reviews:** {df_c['ratings_count'].mean():.0f}  ·  "
                    f"**Cities represented:** {df_c['city'].nunique()}"
                )

                # Rating distribution within cluster
                col_hist, col_top = st.columns([1, 1])
                with col_hist:
                    fig_hist = px.histogram(
                        df_c, x="weighted_rating", nbins=20,
                        labels={"weighted_rating": "Confidence-weighted Rating", "count": "Venues"},
                        color_discrete_sequence=["#2a9d8f"],
                        height=260,
                        title="Rating distribution",
                    )
                    fig_hist.add_vline(x=df_c["weighted_rating"].mean(), line_dash="dash",
                                       line_color="#e76f51", annotation_text="mean")
                    fig_hist.update_layout(showlegend=False, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_top:
                    st.markdown("**Top 10 venues (by confidence-weighted rating)**")
                    top_cols = ["name", "city", "rating", "weighted_rating", "price_nis", "ratings_count"]
                    if "google_maps_url" in df_c.columns:
                        top_cols.append("google_maps_url")
                    top10 = df_c.nlargest(10, "weighted_rating")[top_cols].copy()
                    top10["weighted_rating"] = top10["weighted_rating"].round(2)
                    top10["price_nis"] = top10["price_nis"].round(0).astype(int)
                    top10.index = range(1, len(top10) + 1)
                    st.dataframe(
                        top10.rename(columns={
                            "name": "Venue", "city": "City",
                            "rating": "Raw ⭐", "weighted_rating": "Conf. ⭐",
                            "price_nis": "₪", "ratings_count": "Reviews",
                            "google_maps_url": "Maps",
                        }),
                        use_container_width=True,
                        column_config={
                            "Raw ⭐": st.column_config.NumberColumn(format="%.1f"),
                            "Conf. ⭐": st.column_config.NumberColumn(format="%.2f"),
                            "Maps": st.column_config.LinkColumn("Maps", display_text="Open ↗"),
                        },
                    )


# ══════════════════════════════════════════════════════════════
# TAB 6 — Agent (M4): free-text profile → LLM extracts params → YOUR model ranks
# ══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 Agent — From Free Text to Recommendation")
    st.caption(
        "Describe yourself in your own words. A Groq/Llama LLM reads your text and classifies it "
        "into **4 parameters** (city · budget · quality · user type); then **your KMeans model** "
        "clusters and ranks venues and returns the **5 optimal** ones. "
        "The numbers always come from the model — the LLM only translates input and output."
    )

    km_saved = load_model()
    if km_saved is None:
        st.warning("No trained model found. Open the **🔮 Predicted** tab and click *Train & Compare* first.")
        st.stop()

    cities_avail = sorted(df_all["city"].dropna().unique().tolist())

    # st.secrets raises (not returns) when no secrets.toml exists anywhere — guard it.
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

    st.divider()
    st.markdown("#### Step 1 — Describe yourself")
    user_text = st.text_area(
        "Tell me about you and what you're after — age, life stage, budget, how much quality "
        "matters, and where you are. Free text, Hebrew or English.",
        value='אני סטודנט בן 23 בחיפה, התקציב שלי עד 50 ש"ח, ואני מחפש את האיכות הכי גבוהה שאפשר.',
        height=130,
        help="The LLM extracts the 4 parameters (city · budget · quality · user type) from this text.",
    )

    if not api_key:
        st.info(
            "No `GROQ_API_KEY` in `st.secrets`, so free-text extraction is disabled. Add a free key "
            "to `.streamlit/secrets.toml` to enable the LLM, or use the city fallback below."
        )

    st.divider()
    st.markdown("#### Step 2 — Run the agent")
    if st.button("🥙 Find my shawarma", type="primary"):
        params = None
        if api_key and user_text.strip():
            with st.spinner("LLM reading your text and extracting parameters…"):
                params = extract_params(user_text, api_key, cities_avail)
        st.session_state["agent_params"] = params
        st.session_state["agent_ran"] = True

    if st.session_state.get("agent_ran"):
        params = st.session_state.get("agent_params")
        df_top = pd.DataFrame()

        if params is not None:
            st.markdown("##### Extracted parameters (the 4 categories)")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("מיקום · City", params["city"] or "—")
            pc2.metric("סכום · Budget", f"₪{params['max_budget_nis']}")
            pc3.metric("איכות · Quality", params["quality_preference"])
            pc4.metric("סוג · User type", params["user_type"])
            df_top = recommend(km_saved, df_all, params, top_n=5)
        elif api_key:
            st.warning("The LLM couldn't produce valid parameters from your text — using the city fallback.")

        st.divider()
        if params is not None and not df_top.empty:
            target_label = df_top.attrs.get("model_target_label")
            if target_label:
                st.success(
                    f"🧠 **Your KMeans model** predicted the cluster for this profile → "
                    f"**«{target_label}»** ({df_top.attrs.get('model_cluster_n', 0)} venues in it). "
                    "The LLM only extracted the 4 parameters above — the model chose the group below."
                )
            st.markdown(f"#### 🏆 Top {len(df_top)} venues for you")
            disp = ["name", "city", "rating", "weighted_rating", "price_nis", "cluster_label", "score"]
            if "google_maps_url" in df_top.columns:
                disp.append("google_maps_url")
            out = df_top[disp].copy()
            out["weighted_rating"] = out["weighted_rating"].round(2)
            out["score"] = out["score"].round(2)
            out.index = range(1, len(out) + 1)
            st.dataframe(
                out.rename(columns={
                    "name": "Venue", "city": "City", "rating": "Rating",
                    "weighted_rating": "Conf. ⭐", "price_nis": "Price (NIS)",
                    "cluster_label": "Model cluster", "score": "Score",
                    "google_maps_url": "Maps",
                }),
                use_container_width=True,
                column_config={
                    "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
                    "Conf. ⭐": st.column_config.NumberColumn(format="%.2f"),
                    "Price (NIS)": st.column_config.NumberColumn(format="₪%.0f"),
                    "Maps": st.column_config.LinkColumn("Maps", display_text="Open ↗"),
                },
            )
        else:
            # Fallback path (point 3): ask only for the city, return best / cheapest / closest.
            st.markdown("#### Fallback — tell me your city")
            st.caption("I'll give you three solid picks: the best, the cheapest, and the closest.")
            fb_city = st.selectbox("Which city are you in?", cities_avail, key="agent_fb_city")
            picks = fallback_recommend(df_all, fb_city)
            if not picks:
                st.error("No venues found for that city.")
            else:
                fc1, fc2, fc3 = st.columns(3)
                labels = [("🏅 Best", "best"), ("🪙 Cheapest", "cheapest"), ("📍 Closest", "closest")]
                for col, (title, key) in zip((fc1, fc2, fc3), labels):
                    row = picks[key].iloc[0]
                    with col:
                        st.markdown(f"**{title}**")
                        st.markdown(f"**{row['name']}**")
                        st.caption(
                            f"⭐ {row['rating']:.1f} · ₪{row['price_nis']:.0f} · {row['distance_km']:.2f} km"
                        )
