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
- [x] Normalize features with `StandardScaler` on `[price_nis, weighted_rating]`
- [x] Train K-Means with k=9 fixed (one cluster per target class)
- [x] Train DBSCAN (eps auto-tuned, min_samples=5) with KNN fallback for test evaluation
- [x] Train Agglomerative Clustering (ward linkage, k=9)
- [x] 80/20 train/test split — report Silhouette Score on both splits
- [x] KMeans test silhouette ~0.37 · DBSCAN ~0.71 · Agglomerative ~0.37
- [x] 9-class confusion matrix via Hungarian cluster→class matching
- [x] Implement persona-weighted ranking score (student / quality personas)
- [x] Add 🔮 Predicted tab: Train & Compare → KPI table → confusion matrix → per-cluster deep-dive
- [x] Save/load model via `data/kmeans_model.pkl`
- [x] Write `tests/test_smoke.py` (8 tests, all passing)
- [x] Merge `Eliad` branch into `main`

## M4 · Agent Layer (model → agent, input layer) ✅ (completed — 2026-06-30)
- [x] Build `src/agent.py` wrapping the M3 model (the LLM translates; the model decides)
- [x] System + user prompts; Groq/Llama (`llama-3.3-70b-versatile`, `temperature=0`, OpenAI-compatible)
- [x] LLM classifies free text into 4 params: city · budget · quality · user_type (strict JSON)
- [x] Output validation + Fallback (ask for city → best / cheapest / closest)
- [x] Model-driven selection: `recommend()` calls `model.predict()` on the user's ideal point to pick the cluster, then returns venues from that cluster (top-5)
- [x] Step 4 — `phrase_response()`: LLM restates the model's picks in the user's language (no inventing)
- [x] 🤖 Agent tab with free-text input; key in `st.secrets` (`.streamlit/secrets.toml`, git-ignored)
- [x] Remove 🗺️ Haifa tab; drop unused `folium` / `streamlit-folium` deps
- [x] Deploy-prep: pin numpy/scipy, retrain model under sklearn 1.5.2

## M5 · Evaluation & Presentation 🔲 (weeks 8–9)
- [ ] Final Silhouette Score report
- [ ] Manual persona match evaluation (50 test cases)
- [ ] Response time benchmark (< 3 sec GPS → list)
- [ ] Presentation slides + live demo
