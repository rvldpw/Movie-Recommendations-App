# 🎬 CineWrap 2026: Hybrid Movie Recommender

A portfolio-grade **hybrid recommendation system** built with Python and Streamlit, inspired by *Netflix Wrapped*, designed to deliver personalized movie recommendations through collaborative filtering, content similarity, and popularity signals.

---

## Project Overview

This project combines **machine learning recommendation techniques** with business-style ranking logic to generate more relevant movie recommendations.

Instead of relying on one recommendation method alone, the system blends:
- **Collaborative Filtering** to learn user taste patterns  
- **Content Similarity** to recommend similar movies  
- **Popularity Signals** to improve quality and discovery

This hybrid approach helps balance **personalization, cold-start handling, and recommendation reliability.**

---

## Features

- **Collaborative Filtering (SVD)** — Learns hidden user–movie preference patterns through matrix factorization  
- **Content-Based Filtering** — Uses genre cosine similarity to find similar titles  
- **Popularity Blending** — Balances personalization with well-rated movies  
- **Recency Weighting** — Recent preferences carry stronger influence  
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

---

## What is SVD?

**Singular Value Decomposition (SVD)** is a matrix factorization method commonly used in recommender systems.

Rather than relying only on explicit ratings, SVD learns **latent preference patterns** such as hidden user tastes and movie characteristics, then predicts what a user is likely to enjoy.

In this project, SVD powers the **collaborative filtering engine**, which serves as the strongest personalization signal in the hybrid model.

---

## Why This Approach?

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
