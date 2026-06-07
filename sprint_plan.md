# Sprint Plan — Appetite Engineering

## M1 · Data Card & Scope ✅ (completed)
- [x] Define problem statement and personas
- [x] Identify data sources (Google Maps API + price scrape)
- [x] Write Data Card (fields, biases, gaps)
- [x] Create repository structure
- [x] Write README + CLAUDE.md

## M2 · EDA + Data Dashboard ✅ (completed)
- [x] Collect & clean full dataset (12,270 venues, 77 cities)
- [x] Store clean data in `data/dataset.csv`
- [x] Build `src/data.py` with `load_raw()`, `clean()`, `build_features()`
- [x] Build `src/eda.py` with metric and chart computation functions
- [x] EDA tab in Streamlit: 4 metrics, 3 charts, sidebar filters, map, insights
- [x] Interactive sidebar: product selector, city multiselect, price slider, rating slider, parking checkbox
- [x] Map visualization on OpenStreetMap (lat/lng color-coded by rating)
- [x] Update README with run commands, data description, EDA insights, screenshot
- [x] Update CLAUDE.md with actual data findings

## M3 · Clustering Model ✅ (completed — 2026-06-07)
- [x] Normalize features with `StandardScaler` on `[price_nis, rating, reviews_count]`
- [x] Train K-Means with k auto-tuned ∈ {3…8} via Silhouette sweep (best k=4)
- [x] Train DBSCAN (eps=0.5, min_samples=5) with KNN fallback for test evaluation
- [x] Train Agglomerative Clustering (ward linkage, same k sweep)
- [x] 70/30 train/test split — report Silhouette Score on both splits
- [x] KMeans test silhouette = 0.373 · DBSCAN = 0.708 · Agglomerative = ~0.37
- [x] Implement persona-weighted ranking score (student / quality personas)
- [x] Add 🤖 Recommend tab: city selector → Train & Compare → elbow chart + KPI table → ranked venues
- [x] Save/load model via `data/kmeans_model.pkl`
- [x] Write `tests/test_smoke.py` (8 tests, all passing)
- [x] Merge `Eliad` branch into `main`

## M4 · Evaluation & Presentation 🔲 (weeks 8–9)
- [ ] Final Silhouette Score report
- [ ] Manual persona match evaluation (50 test cases)
- [ ] Response time benchmark (< 3 sec GPS → list)
- [ ] Finalize Streamlit UI with all tabs
- [ ] Presentation slides + live demo
