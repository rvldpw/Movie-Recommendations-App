# 🎬 MovieWrap — Hybrid Movie Recommender

A portfolio-grade hybrid recommender system built with Python + Streamlit, inspired by "Netflix Wrapped".

## Features

- **Content-based filtering** — genre profile cosine similarity
- **Collaborative filtering** — SVD-based user–user similarity
- **Popularity blending** — ensures well-known movies aren't buried
- **Recency weighting** — recent ratings matter more
- **Streamlit UI** — interactive, sidebar controls, expandable recommendations

## Hybrid Scoring

| Signal | Weight |
|---|---|
| Collaborative (SVD cosine) | 40% |
| Content (genre cosine) | 35% |
| Popularity (avg rating) | 25% |

## Project Structure

```
movie-recommender/
├── app.py                  # Streamlit app entry point
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── data_sample.csv     # Your ratings dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # CSV ingestion + datetime parsing
│   ├── feature_engineering.py  # Recency weighting, genre profiles
│   ├── collaborative.py    # SVD collaborative engine
│   └── recommender.py      # Hybrid RecommenderSystem class
└── models/                 # (optional) persisted model artefacts
```

## Dataset Format

Your CSV must contain at minimum:

| Column | Description |
|---|---|
| `userId` | Integer user identifier |
| `movieId` | Integer movie identifier |
| `title` | Movie title string |
| `rating` | Numeric rating (e.g. 0.5–5.0) |
| `timestamp` | Unix timestamp (seconds) |
| `Action`, `Drama`, … | One-hot genre columns |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
cp /path/to/your/data.csv data/data_sample.csv

# 4. Run
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub (see what **not** to upload below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at `app.py`.
4. Deploy — done.

> **Note:** If your dataset is large, reduce it to ≤ 10 MB before committing, or load it from a URL.

## What NOT to Upload

- Raw huge datasets (> 50 MB) — use a reduced sample
- `*.pkl` model files — regenerated on startup
- `__pycache__/` or `.ipynb_checkpoints/`

## Future Upgrades

- **Cold-start fallback** — popular movies for new users
- **Explainability** — "Because you liked *Interstellar*…"
- **Diversity re-ranking** — avoid 10 recommendations in the same genre
- **Evaluation metrics** — Precision@K, Recall@K, MAP@K
