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

## M3 · Clustering Model 🔲 (next — weeks 6–7)
- [ ] Normalize features with `StandardScaler` on `[price_nis, rating, distance_km, ratings_count]`
- [ ] Train K-Means with k sweep ∈ {3…8} using Elbow + Silhouette
- [ ] Evaluate Silhouette Score on 15% test split (target ≥ 0.45)
- [ ] Implement persona-weighted ranking score per cluster
- [ ] Add "Recommend" tab to Streamlit: GPS input → ranked venue list
- [ ] Beat baseline (naïve distance sort)

## M4 · Evaluation & Presentation 🔲 (weeks 8–9)
- [ ] Final Silhouette Score report
- [ ] Manual persona match evaluation (50 test cases)
- [ ] Response time benchmark (< 3 sec GPS → list)
- [ ] Finalize Streamlit UI with all tabs
- [ ] Presentation slides + live demo
