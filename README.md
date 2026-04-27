# 🎬 CineWrap 2026: Hybrid Movie Recommender

A portfolio-grade **hybrid recommendation system** built with Python and Streamlit, inspired by *Netflix Wrapped*, designed to deliver personalized movie recommendations through collaborative filtering, content similarity, and popularity signals.

---

## Features

- **Collaborative Filtering** — SVD-based user similarity recommendations  
- **Content-Based Filtering** — Genre cosine similarity matching  
- **Popularity Blending** — Balances personalization with highly rated titles  
- **Recency Weighting** — Recent user preferences carry stronger influence  
- **Interactive Streamlit App** — User controls and recommendation explorer  

---

## 🤖 Hybrid Scoring Model

Recommendations are ranked using a weighted hybrid score:

| Signal | Weight |
|---|---|
| Collaborative Filtering (SVD Cosine) | 50% |
| Content Similarity (Genre Cosine) | 30% |
| Popularity Signal (Average Rating) | 20% |

### Final Score
**Hybrid Score =**  
`0.50 × Collaborative Score`  
`+ 0.30 × Content Similarity`  
`+ 0.20 × Popularity Score`

### Why This Approach?
- Prioritizes **personalization** through collaborative filtering  
- Uses **content similarity** to support cold-start recommendations  
- Adds **popularity** as a stabilizing quality signal  
- Balances relevance, diversity, and discovery

---

## Project Structure

```text
movie-recommender/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── collaborative.py
│   └── recommender.py
└── models/
```

---

## Dataset Format

| Column | Description |
|---|---|
| `userId` | User identifier |
| `movieId` | Movie identifier |
| `title` | Movie title |
| `rating` | User rating |
| `timestamp` | Rating timestamp |
| `Action`, `Drama`, ... | One-hot encoded genres |
