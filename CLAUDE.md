# CLAUDE.md — Appetite Engineering

## Project Goal (one line)
ML-powered app that recommends the best shawarma venue near the user by clustering venues on price, rating, and distance — personalized per user persona.

## Data Sources
- **Google Maps Places API** (`googlemaps` Python SDK) — fields: `place_id`, `name`, `lat`, `lng`, `rating`, `user_ratings_total`, `price_level`
- **Supplementary price scrape** — exact NIS prices scraped from Wolt / venue sites, stored in `data/prices.csv`
- Raw API responses are cached in `data/raw/` and never committed. Processed features live in `data/processed/`.

## What We Learned About the Data (M2)
- **Dataset:** 12,270 unique venues across 77 Israeli cities after deduplication on `name + lat + lng`
- **Price range:** Turkey shawarma pita ₪37–₪58, median ₪46, IQR ₪43–₪48 (tight cluster)
- **Rating distribution:** Mean 4.32, median 4.4, std 0.53 — ratings skew high; most venues above 4.0
- **Price-quality correlation:** Pearson r = −0.06 — effectively zero. Price does NOT predict quality.
- **Geographic spread:** Average city price varies by ₪13.6 → city/location is a meaningful feature
- **Price source:** 99.9% of prices are estimated (not scraped from menus) — treat as ordinal signal, not ground truth
- **reviews_count:** now populated for all 774 venues in `data/dataset.csv` (backfilled from Google review counts), so `weighted_rating` (Bayesian, confidence-weighted by review count) is a meaningful, non-constant signal — it is used as the model's quality feature (see M3 below), not just shown for reference.
- **Coordinate columns:** Raw CSV uses `latitude`/`longitude`; `src/data.py` renames them to `lat`/`lng`

## ML Model (M3 — implemented)

### Algorithms trained & compared (80 / 20 train/test split, `random_state=42`)
| Algorithm | Paradigm | Train Silhouette | Test Silhouette | predict() |
|-----------|----------|-----------------|----------------|-----------|
| **KMeans** (k=9, fixed) | Partitional | ~0.59 | ~0.58 | ✅ native |
| DBSCAN (eps auto-tuned, min_samples=5) | Density-based | ~1.00 | ~0.68 | ❌ KNN fallback |
| Agglomerative (ward, k=9) | Hierarchical | ~0.57 | ~0.57 | ❌ KNN fallback |

- **Selected model:** KMeans — only algorithm with native `predict()` for new venues
- **Input features:** `[price_nis, weighted_rating]` — StandardScaler-normalized. `weighted_rating` (Bayesian-smoothed by `reviews_count`, pulling low-review-count venues toward the global mean) replaced raw `rating` as the quality feature once `reviews_count` was backfilled — raw `rating` had let a venue with 3 five-star reviews rank identically to one with 3,000. Raw `rating` is still shown alongside for reference. Distance is computed at query time, not used in clustering.
- **k is fixed at 9** — one cluster per target class: {good / average / bad} × {expensive / reasonable / affordable}. Rating bands (on `weighted_rating`): good ≥ 4.5, average 4.0–4.4, bad < 4.0. Price bands are fixed NIS thresholds: expensive > ₪60, reasonable ₪54–60, affordable < ₪54 (`price_nis` is the row-wise mean of the 4 shawarma price columns: turkey/cow × pita/laffa). Clusters are mapped to those 9 classes via Hungarian matching for a confusion-matrix accuracy readout, which is always a fixed 9×9 (using the full class list, not just classes observed in a given train/test split).
- **Output:** cluster label per venue + persona-weighted ranking score
- **Success metric:** Silhouette Score ≥ 0.45 on test split + 9-class confusion-matrix accuracy. DBSCAN's silhouette is inflated (computed on non-noise points only) and it can't generalize without a KNN fallback — hence KMeans is still selected. (Benchmark numbers in the table above predate the current dataset/feature swap and should be re-measured via the Train & Compare button.)
- **Baseline to beat:** random cluster assignment (same k=9), computed per run
- **Saved model:** `data/kmeans_model.pkl` (auto-loaded by Streamlit on next run)

## Agent Layer (M4 — implemented)
The agent **wraps** the M3 model; it never invents numbers — the LLM translates input and output, the model decides. Four-step loop:
1. User describes themselves in **free text** (Hebrew or English) in the 🤖 Agent tab.
2. A Groq/Llama LLM (`llama-3.3-70b-versatile`, `temperature=0`) classifies the text into **4 parameters** as strict JSON: `city` (מיקום), `max_budget_nis` (סכום), `quality_preference` (איכות), `user_type` (סוג משתמש). Invalid output (bad JSON / out-of-range / unknown city) → fallback: ask only for the city and return **best / cheapest / closest**.
3. **The model selects.** `recommend()` builds the user's *ideal venue* from real data quantiles and calls `model.predict()` on it; the cluster the model assigns to that ideal is the target, and the agent returns real venues the model placed in that same cluster. Persona weights only **order within** the model-chosen cluster — they never override which cluster the model picked. → **top 5**.
4. **The LLM phrases the model's output** (`phrase_response`, `temperature=0.3`): it is handed the exact venues the model chose and may only restate them in the user's language — never invent, add, drop, or change a venue/price/rating. Rendered as a chat reply above the evidence table.
- Groq is **OpenAI-compatible, not Anthropic** — use `client.chat.completions.create`, not `messages.create`.
- Quality filtering and the ideal-venue point both use `weighted_rating`, matching the model's own feature.
- **Distance always ranks against a real point** — never a constant 0. If the LLM-extracted address geocodes, that point is used; otherwise `city_center()` (mean lat/lng of the matched city's venues) is used as a stand-in. `PERSONA_WEIGHTS[...]["distance_km"]` is negative, so farther venues score lower within the model-chosen cluster. `recommend()`/`fallback_recommend()` expose `user_lat`/`user_lng`/`location_source` (`"geocoded"` / `"city_center"` / `"none"`) so the UI can plot the assumed user location alongside the venues.
- Key lives in `st.secrets["GROQ_API_KEY"]` via `.streamlit/secrets.toml` (git-ignored). Without a key the tab uses the city fallback only.

## File Structure
```
app.py                 — Streamlit entry point (6 tabs); calls src/ modules only, no logic here
src/data.py            — load_raw(), clean() (adds price_nis, weighted_rating), build_features()
src/eda.py             — EDA chart functions (used by EDA tab)
src/model.py           — split_data(), find_best_k(), train_kmeans(), train_dbscan(),
                         train_agglomerative(), compare_algorithms(),
                         compute_confusion_matrix(), assign_cluster_labels(),
                         save_model(), load_model(), predict()
src/agent.py           — M4 agent: build_system_prompt(), build_user_prompt(),
                         extract_params() (Groq LLM), validate_params(),
                         recommend() (model picks cluster → top-5), fallback_recommend(),
                         phrase_response() (LLM restates the model's picks)
data/dataset.csv       — 12,270 clean venues (committed)
data/kmeans_model.pkl  — trained KMeans model (committed; regenerate with Train button)
.streamlit/secrets.toml          — GROQ_API_KEY (git-ignored; never commit)
.streamlit/secrets.toml.example  — template (committed)
tests/test_core.py     — core tests; must all pass on every commit
notebooks/             — EDA only; no production code lives here
```

The 6 Streamlit tabs: 🎯 Problem & Personas · 📚 Literature & Market · 📊 EDA · 🏆 KPI & Model · 🔮 Predicted · 🤖 Agent.

## Coding Conventions
- **Language:** English for all code, variable names, and comments
- **Comments:** only when the WHY is non-obvious; no docstrings unless a function is exported
- **Variable names:** `snake_case`; DataFrames prefixed `df_`; model artifacts suffixed `_model`
- **No hardcoded paths** — use `pathlib.Path` relative to project root
- **Secrets** — API keys from `st.secrets` (`.streamlit/secrets.toml`, git-ignored) or `os.environ`; never in code, notebooks, or commits
- **Streamlit state** — use `st.session_state` for anything that must survive reruns

## What NOT to do
- Do not put business logic in `app.py` — it is a thin UI shell
- Do not commit `data/raw/`, `.env`, or any file > 50 MB
- Do not add features beyond the current milestone scope without asking first
