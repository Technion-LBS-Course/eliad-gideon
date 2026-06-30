# Appetite Engineering — הנדסת התיאבון
**Real-time shawarma navigation: ML-powered recommendations that balance quality, price, and distance in a single tap.**

> **KPI:** Silhouette Score ≥ 0.45 on the held-out venue set — the only objective measure of cluster quality when there are no ground-truth labels.

---

## How to Run

```bash
git clone https://github.com/gidihoresh13/eliad-gideon.git
pip install -r requirements.txt
streamlit run app.py
```

**Optional — enable the M4 agent (free-text recommendations):** add a free [Groq](https://console.groq.com) key so the 🤖 Agent tab can run the LLM. Copy the template and drop your key in:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml → GROQ_API_KEY = "gsk_..."
```

`.streamlit/secrets.toml` is git-ignored and must never be committed. Without a key the Agent tab still works via the city fallback (best / cheapest / closest).

---

## Dashboard Screenshot

![Dashboard](docs/screenshot.png)

---

## Data Description

| Property | Value |
|----------|-------|
| **File** | `data/dataset.csv` |
| **Rows after cleaning** | 12,270 venues |
| **Columns** | 20 |
| **Cities** | 77 Israeli cities |
| **Source** | Google Maps Places API + estimated price fills |
| **Coordinates** | `lat`, `lng` (float, valid Israeli range) |
| **Missing ratings** | 485 venues (3.9%) — kept; filtered in EDA |
| **Duplicates removed** | Rows sharing identical `name + lat + lng` |
| **Price source** | 99.9% estimated from price_level ordinal; flagged in `price_source` column |

**Columns:** `latitude`, `longitude`, `city`, `name`, `rating`, `reviews_count`, six price columns (turkey/cow/falafel × pita/laffa), `price_small_fries`, `price_big_fries`, `price_drink`, `car_park_nearby`, `price_source`, `price_notes`, `google_maps_url`

**Variable types:** lat/lng → float64; rating, reviews_count, all price columns → float64/int; car_park_nearby → bool; city, name, price_source → object

---

## EDA Insights

1. **Price has almost no correlation with quality** — Pearson r = −0.06 between turkey shawarma pita price and rating. A ₪58 shawarma is statistically no better than a ₪44 one. This confirms that ranking by price alone is meaningless and validates the need for multi-dimensional clustering.

2. **Prices cluster in a narrow ₪43–₪48 IQR band** — 50% of venues fall within a 5 NIS window (median ₪46). Price is a poor standalone differentiator; rating and distance must be weighted in to produce meaningful venue separation.

3. **Geographic arbitrage exists: ₪13.6 spread across cities** — Average turkey pita price varies by ₪13.6 between the cheapest and most expensive cities. Distance + city context is a meaningful signal, justifying `distance_km` as a core model feature alongside price.

---

## The Problem

Israel's shawarma market has 12,000+ venues with prices ranging ₪37–₪58 per portion, yet there is **no direct correlation between price and quality**. A hungry user at lunchtime must simultaneously weigh three competing constraints: distance (current hunger), price (budget), and rating (quality expectations).

- **Who hurts:** Students, office workers, anyone making a spontaneous lunch decision under time pressure.
- **Why digital helps:** The data exists (Google Maps ratings, venue coordinates, scraped prices) but is scattered, unordered, and not personalized.
- **The gap:** Existing apps optimize for delivery revenue, not the user's value profile.

---

## Target Audience

**Primary Persona — הסטודנט החסכן (The Thrifty Student)**
- Age 20–28, tight budget (~45–52 NIS ceiling), values quantity and taste over prestige.
- Algorithm weights: Price ×1.5 · Rating ×1.0 · Distance ×0.8

**Secondary Persona — חובב האיכות (The Quality Enthusiast)**
- Willing to travel further and pay up to 60 NIS for a 4.8+ rated experience.
- Algorithm weights: Rating ×2.0 · Price ×0.5 · Distance ×0.5

---

## M3 — Algorithm Training & Comparison

### Algorithms compared

| Algorithm | Paradigm | Hyperparameters | Train Silhouette | Test Silhouette | Meets KPI ≥ 0.45 |
|-----------|----------|----------------|-----------------|----------------|-----------------|
| **KMeans** | Partitional | k=9 (fixed — one per target class) | ~0.37 | ~0.37 | ❌ |
| **DBSCAN** | Density-based | eps auto-tuned, min_samples=5 | ~0.73 | ~0.71 | ✅ |
| **Agglomerative** | Hierarchical | k=9 (ward linkage) | ~0.37 | ~0.37 | ❌ |

**Selected model: KMeans (k=9)** — only algorithm with native `predict()` for new venues. DBSCAN achieves a higher silhouette but cannot generalize out-of-sample without a KNN fallback (and its score is inflated by being computed on non-noise points only). The low KMeans silhouette reflects the dataset's narrow ₪5 IQR price band, which limits cluster separability. k is fixed at 9 so each cluster maps to one target class — {good / medium / bad score} × {high / fair / low price} — enabling a confusion-matrix accuracy readout via Hungarian matching.

### Train / Test Split
- **Split:** 80% train / 20% test, `random_state=42`
- **Eligible rows:** venues with non-null `price_nis` + `rating` (~12k venues)
- **Features:** `[price_nis, weighted_rating]` — StandardScaler-normalized. `weighted_rating` is a Bayesian confidence-smoothed rating (low-review venues are pulled toward the global mean).
- **KPI:** Silhouette Score ≥ 0.45 on test split + 9-class confusion-matrix accuracy

### Running the ML pipeline in Streamlit
1. Open the **🔮 Predicted** tab
2. Click **🔬 Train & Compare Models** — see the train/test KPI table, confusion matrix, and per-cluster deep-dive
3. The trained model is persisted at `data/kmeans_model.pkl` and auto-loaded on next run

---

## M4 — Agent Layer (free-text → model)

The agent **wraps** the M3 model — it translates language, the model produces the numbers. Loop:

1. **Free text in** — in the **🤖 Agent** tab you describe yourself in your own words (Hebrew or English), e.g. *"אני סטודנט בן 23 בחיפה, תקציב עד 50 ש"ח, רוצה את האיכות הכי גבוהה"*.
2. **LLM extracts 4 parameters** — a Groq/Llama model (`llama-3.3-70b-versatile`, `temperature=0`) returns strict JSON: `city` (מיקום) · `max_budget_nis` (סכום) · `quality_preference` (איכות) · `user_type` (סוג משתמש).
3. **Your model ranks** — the saved KMeans model clusters the matching venues and ranks them by persona-weighted score → **top 5**.
4. **Fallback** — if the LLM output is invalid (bad JSON / out of range / unknown city), the agent asks only for the city and returns **best / cheapest / closest**.

Implemented in `src/agent.py`. Groq is **OpenAI-compatible** (`chat.completions.create`). The key is read from `st.secrets["GROQ_API_KEY"]` (see *How to Run*).

---

## Formal ML Problem Statement

| Component | Definition |
|-----------|-----------|
| **Input X** | `[price_nis, weighted_rating]` — StandardScaler-normalized venue features |
| **Output y** | 9-class label: {good / medium / bad score} × {high / fair / low price}; plus persona-weighted ranking score |
| **Algorithm** | K-Means · k = 9 (fixed — one cluster per target class) |
| **Loss / Objective** | Minimize intra-cluster variance; maximize inter-cluster separation |
| **Train / Test** | 80% / 20% random split, `random_state=42` |
| **Distance** | Haversine-computed at query time; used for filtering + scoring, not clustering |
| **Baseline** | Random cluster assignment (same k = 9), computed per run |

---

## File Structure

```
app.py                      — Streamlit entry point (6 tabs; UI shell only, no logic)
requirements.txt            — Pinned dependencies
README.md                   — This file
sprint_plan.md              — Milestone tracker
CLAUDE.md                   — Coding conventions and project context
.streamlit/
├── secrets.toml            — GROQ_API_KEY (git-ignored; never commit)
└── secrets.toml.example    — Template (committed)
data/
├── dataset.csv             — 12,270 clean venues (committed)
└── kmeans_model.pkl        — Trained KMeans model (auto-loaded by app)
src/
├── __init__.py
├── data.py                 — load_raw(), clean() [adds price_nis, weighted_rating], build_features()
├── eda.py                  — EDA chart functions
├── model.py                — split_data(), find_best_k(), train_kmeans(),
│                             train_dbscan(), train_agglomerative(), compare_algorithms(),
│                             compute_confusion_matrix(), save_model(), load_model(), predict()
└── agent.py                — M4 agent: build_system_prompt(), build_user_prompt(),
                              extract_params() [Groq LLM], validate_params(),
                              recommend() [top-5 via model], fallback_recommend()
tests/
└── test_smoke.py           — 8 smoke tests (all must pass on every commit)
notebooks/
└── 01_eda.ipynb            — Exploratory analysis
.gitignore
```

**Tabs:** 🎯 Problem & Personas · 📚 Literature & Market · 📊 EDA · 🏆 KPI & Model · 🔮 Predicted · 🤖 Agent

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | ~99% of prices are estimated, not verified | High | Medium | Flag in UI with `price_source` badge; use as ranking signal, not absolute truth |
| 2 | Clustering instability — optimal k unclear | Medium | Medium | Sweep k ∈ {3…8} with Elbow + Silhouette; fall back to DBSCAN |
| 3 | Google Maps API quota overrun | Medium | Low | All responses cached in `data/raw/`; throttle to 1 req/s |
