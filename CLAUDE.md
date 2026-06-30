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

### Algorithms trained & compared (80 / 20 train/test split, `random_state=42`)
| Algorithm | Paradigm | Train Silhouette | Test Silhouette | predict() |
|-----------|----------|-----------------|----------------|-----------|
| **KMeans** (k=9, fixed) | Partitional | ~0.37 | ~0.37 | ✅ native |
| DBSCAN (eps auto-tuned, min_samples=5) | Density-based | ~0.73 | ~0.71 | ❌ KNN fallback |
| Agglomerative (ward, k=9) | Hierarchical | ~0.37 | ~0.37 | ❌ KNN fallback |

- **Selected model:** KMeans — only algorithm with native `predict()` for new venues
- **Input features:** `[price_nis, weighted_rating]` — StandardScaler-normalized. `weighted_rating` is a Bayesian confidence-smoothed rating (pulled toward the global mean when `reviews_count` is missing). Distance is computed at query time, not used in clustering.
- **k is fixed at 9** — one cluster per target class: {good / medium / bad score} × {high / fair / low price}. Clusters are mapped to those 9 classes via Hungarian matching for a confusion-matrix accuracy readout.
- **Output:** cluster label per venue + persona-weighted ranking score
- **Success metric:** Silhouette Score ≥ 0.45 on test split (DBSCAN meets it; KMeans ~0.37 due to the tight ₪5 IQR price band) + 9-class confusion-matrix accuracy
- **Baseline to beat:** random cluster assignment (same k=9), computed per run
- **Saved model:** `data/kmeans_model.pkl` (auto-loaded by Streamlit on next run)

## Agent Layer (M4 — implemented)
The agent **wraps** the M3 model; it never invents numbers. Four-step loop:
1. User describes themselves in **free text** (Hebrew or English) in the 🤖 Agent tab.
2. A Groq/Llama LLM (`llama-3.3-70b-versatile`, `temperature=0`) classifies the text into **4 parameters** as strict JSON: `city` (מיקום), `max_budget_nis` (סכום), `quality_preference` (איכות), `user_type` (סוג משתמש).
3. The **saved KMeans model** assigns clusters and ranks the surviving venues by persona-weighted score → **top 5**.
4. **Validation + fallback:** invalid LLM output (bad JSON, out-of-range, unknown city) is caught; the agent then asks only for the city and returns three safe picks — **best / cheapest / closest**.
- Groq is **OpenAI-compatible, not Anthropic** — use `client.chat.completions.create`, not `messages.create`.
- Quality filtering uses raw `rating` (not `weighted_rating`, which collapses to the global mean for the ~95% of venues lacking a review count).
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
                         recommend() (top-5 via model), fallback_recommend()
data/dataset.csv       — 12,270 clean venues (committed)
data/kmeans_model.pkl  — trained KMeans model (committed; regenerate with Train button)
.streamlit/secrets.toml          — GROQ_API_KEY (git-ignored; never commit)
.streamlit/secrets.toml.example  — template (committed)
tests/test_smoke.py    — 8 smoke tests; must all pass on every commit
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
