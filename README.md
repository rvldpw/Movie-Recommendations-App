# 🎬 MovieWrap: Hybrid Movie Recommender

A portfolio-grade hybrid recommender system built with Python + Streamlit, inspired by "Netflix Wrapped".

## Features

- **Content-based filtering**: genre profile cosine similarity
- **Collaborative filtering**: SVD-based user–user similarity
- **Popularity blending**: ensures well-known movies aren't buried
- **Recency weighting**: recent ratings matter more
- **Streamlit UI**: interactive, sidebar controls, expandable recommendations

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
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # CSV ingestion + datetime parsing
│   ├── feature_engineering.py  # Recency weighting, genre profiles
│   ├── collaborative.py    # SVD collaborative engine
│   └── recommender.py      # Hybrid RecommenderSystem class
└── models/                 # (optional) persisted model artefacts
```

## Dataset Format

Dataset contain:

| Column | Description |
|---|---|
| `userId` | Integer user identifier |
| `movieId` | Integer movie identifier |
| `title` | Movie title string |
| `rating` | Numeric rating (e.g. 0.5–5.0) |
| `timestamp` | Unix timestamp (seconds) |
| `Action`, `Drama`, … | One-hot genre columns |
