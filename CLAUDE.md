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
- **Missing reviews_count:** Most rows have NaN for `reviews_count`; `rating` is available for 96.1%
- **Coordinate columns:** Raw CSV uses `latitude`/`longitude`; `src/data.py` renames them to `lat`/`lng`

## ML Model (M3 — implemented)

### Algorithms trained & compared (70 / 30 train/test split)
| Algorithm | Paradigm | Train Silhouette | Test Silhouette | predict() |
|-----------|----------|-----------------|----------------|-----------|
| **KMeans** (k=4, auto-tuned) | Partitional | 0.377 | 0.373 | ✅ native |
| DBSCAN (eps=0.5, min_samples=5) | Density-based | 0.729 | 0.708 | ❌ KNN fallback |
| Agglomerative (ward, k=4) | Hierarchical | ~0.37 | ~0.37 | ❌ KNN fallback |

- **Selected model:** KMeans — only algorithm with native `predict()` for new venues
- **Input features:** `[price_nis, rating, reviews_count]` — StandardScaler-normalized venue properties (distance computed at query time, not used in clustering)
- **Output:** cluster label per venue + persona-weighted ranking score
- **Success metric:** Silhouette Score ≥ 0.45 on test split (DBSCAN meets it; KMeans at 0.373 due to tight ₪5 IQR price band)
- **Baseline to beat:** naive sort by distance only (no quality/price weighting)
- **Saved model:** `data/kmeans_model.pkl` (auto-loaded by Streamlit on next run)

## File Structure
```
app.py                 — Streamlit entry point (5 tabs); calls src/ modules only, no logic here
src/data.py            — load_raw(), clean() (adds price_nis), build_features()
src/eda.py             — EDA chart functions (used by EDA tab)
src/model.py           — split_data(), find_best_k(), train_kmeans(), train_dbscan(),
                         train_agglomerative(), compare_algorithms(), save_model(),
                         load_model(), predict()
data/dataset.csv       — 12,270 clean venues (committed)
data/kmeans_model.pkl  — trained KMeans model (committed; regenerate with Train button)
tests/test_smoke.py    — 8 smoke tests; must all pass on every commit
notebooks/             — EDA only; no production code lives here
```

## Coding Conventions
- **Language:** English for all code, variable names, and comments
- **Comments:** only when the WHY is non-obvious; no docstrings unless a function is exported
- **Variable names:** `snake_case`; DataFrames prefixed `df_`; model artifacts suffixed `_model`
- **No hardcoded paths** — use `pathlib.Path` relative to project root
- **Secrets** — API keys from `os.environ` only; never in code or notebooks
- **Streamlit state** — use `st.session_state` for anything that must survive reruns

## What NOT to do
- Do not put business logic in `app.py` — it is a thin UI shell
- Do not commit `data/raw/`, `.env`, or any file > 50 MB
- Do not add features beyond the current milestone scope without asking first
