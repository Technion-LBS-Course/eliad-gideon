# CLAUDE.md — Appetite Engineering

## Project Goal (one line)
ML-powered app that recommends the best shawarma venue near the user by clustering venues on price, rating, and distance — personalized per user persona.

## Data Sources
- **Google Maps Places API** (`googlemaps` Python SDK) — fields: `place_id`, `name`, `lat`, `lng`, `rating`, `user_ratings_total`, `price_level`
- **Supplementary price scrape** — exact NIS prices scraped from Wolt / venue sites, stored in `data/prices.csv`
- Raw API responses are cached in `data/raw/` and never committed. Processed features live in `data/processed/`.

## Planned ML Model
- **Algorithm:** K-Means clustering (k tuned via Elbow + Silhouette on validation set)
- **Input features:** `[price_nis, rating, distance_km, ratings_count]` — all StandardScaler-normalized
- **Output:** cluster label per venue + persona-weighted ranking score
- **Success metric:** Silhouette Score ≥ 0.45 on test split
- **Baseline to beat:** naive sort by distance only (no quality/price weighting)

## File Structure
```
app.py          — Streamlit entry point; calls src/ modules only, no logic here
src/data.py     — load_raw(), clean(), build_features() → returns pd.DataFrame
src/model.py    — train(df) → fitted KMeans; predict(model, user_context) → ranked list
notebooks/      — EDA only; no production code lives here
tests/          — pytest; at minimum test_smoke.py must pass on every commit
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
