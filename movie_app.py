import streamlit as st
import pandas as pd
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🍿 CineMatch",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  —  FUN / MEME THEME (optimised)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700;800&display=swap');

:root {
    --bg:        #fffbf0;
    --yellow:    #FFD93D;
    --pink:      #FF6B9D;
    --blue:      #4ECDC4;
    --purple:    #A855F7;
    --orange:    #FF8C42;
    --dark:      #1a1a2e;
    --card-bg:   #ffffff;
    --border:    #1a1a2e;
    --radius:    16px;
    --shadow:    4px 4px 0px #1a1a2e;
    --shadow-lg: 6px 6px 0px #1a1a2e;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--dark) !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--yellow) !important;
    border-right: 3px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--dark) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: white !important;
    border: 2px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 3px 3px 0 var(--border) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hero */
.hero-title {
    font-family: 'Fredoka One', cursive;
    font-size: 4rem;
    color: var(--dark);
    text-shadow: 4px 4px 0 var(--pink), 8px 8px 0 var(--yellow);
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Nunito', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--dark);
    background: var(--blue);
    display: inline-block;
    padding: 4px 14px;
    border: 2px solid var(--border);
    border-radius: 8px;
    box-shadow: 3px 3px 0 var(--border);
    margin-bottom: 1.5rem;
}

.section-header {
    font-family: 'Fredoka One', cursive;
    font-size: 1.8rem;
    color: var(--dark);
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header .pill {
    background: var(--pink);
    border: 2px solid var(--border);
    border-radius: 30px;
    padding: 2px 16px;
    font-size: 1rem;
    box-shadow: 3px 3px 0 var(--border);
}

/* Movie cards */
.movie-card {
    background: var(--card-bg);
    border: 2.5px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
    position: relative;
}
.movie-card:hover {
    transform: translate(-3px, -3px);
    box-shadow: 9px 9px 0px var(--border);
}
.card-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
    border-bottom: 2.5px solid var(--border);
}
.card-poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: linear-gradient(135deg, var(--yellow) 0%, var(--orange) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
    border-bottom: 2.5px solid var(--border);
}
.card-rank {
    position: absolute;
    top: 10px;
    left: 10px;
    background: var(--yellow);
    color: var(--dark);
    font-family: 'Fredoka One', cursive;
    font-size: 1rem;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--border);
    box-shadow: 2px 2px 0 var(--border);
}
.card-score {
    position: absolute;
    top: 10px;
    right: 10px;
    background: var(--pink);
    border: 2px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 800;
    color: white;
    box-shadow: 2px 2px 0 var(--border);
}
.card-body { padding: 0.8rem; }
.card-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1rem;
    color: var(--dark);
    margin-bottom: 0.2rem;
    line-height: 1.3;
}
.card-meta {
    font-size: 0.75rem;
    font-weight: 700;
    color: #666;
    margin-bottom: 0.4rem;
}
.card-genre-pill {
    display: inline-block;
    background: var(--blue);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 1px 8px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-right: 3px;
    margin-bottom: 3px;
    box-shadow: 1px 1px 0 var(--border);
}
.card-plot {
    font-size: 0.78rem;
    color: #444;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin: 0.4rem 0;
}
.card-rating {
    font-family: 'Fredoka One', cursive;
    font-size: 0.95rem;
    color: var(--orange);
    margin-top: 0.3rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Search result card */
.search-card {
    background: white;
    border: 2.5px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: var(--shadow);
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}
.search-card:hover {
    transform: translate(-2px, -2px);
    box-shadow: 6px 6px 0 var(--border);
}
.search-poster {
    width: 55px;
    min-width: 55px;
    height: 80px;
    object-fit: cover;
    border-radius: 8px;
    border: 2px solid var(--border);
}
.search-poster-placeholder {
    width: 55px;
    min-width: 55px;
    height: 80px;
    background: var(--yellow);
    border-radius: 8px;
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}
.search-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1rem;
    color: var(--dark);
}
.search-meta { font-size: 0.8rem; color: #666; font-weight: 600; }
.search-genres {
    display: inline-block;
    background: var(--purple);
    color: white;
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 1px 8px;
    font-size: 0.68rem;
    font-weight: 800;
    margin-right: 3px;
    box-shadow: 1px 1px 0 var(--border);
}

/* Buttons */
div[data-testid="stButton"] > button {
    background: var(--pink) !important;
    color: white !important;
    font-family: 'Fredoka One', cursive !important;
    font-size: 1.1rem !important;
    border: 2.5px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
    padding: 0.5rem 1.5rem !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: var(--shadow-lg) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: none !important;
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: white !important;
    border: 2.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--dark) !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 3px 3px 0 var(--border) !important;
}
div[data-testid="stMultiSelect"] > div,
div[data-testid="stSelectbox"] > div {
    background: white !important;
    border: 2.5px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 3px 3px 0 var(--border) !important;
}
.stSlider [data-testid="stSlider"] { accent-color: var(--pink); }

.tag {
    display: inline-block;
    background: var(--purple);
    color: white;
    border: 2px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 800;
    margin-right: 6px;
    margin-bottom: 6px;
    box-shadow: 2px 2px 0 var(--border);
}

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: white;
    border: 2.5px solid var(--border);
    border-radius: 20px;
    box-shadow: var(--shadow-lg);
    margin: 2rem auto;
    max-width: 500px;
}
.empty-state .emoji { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'Fredoka One', cursive;
    color: var(--dark);
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
}

.sidebar-title {
    font-family: 'Fredoka One', cursive;
    font-size: 1.6rem;
    color: var(--dark);
    margin-bottom: 0.1rem;
}
.sidebar-sub {
    font-size: 0.82rem;
    font-weight: 700;
    color: #555;
    margin-bottom: 1rem;
}
.input-label {
    font-size: 0.82rem;
    font-weight: 800;
    color: var(--dark);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    margin-top: 0.8rem;
}

div[data-testid="stTabs"] button {
    font-family: 'Fredoka One', cursive !important;
    font-size: 1rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid var(--pink) !important;
    color: var(--pink) !important;
}
hr { border-color: var(--border) !important; border-width: 2px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # public demo key (replace with yours)

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western",
]

TMDB_GENRE_MAP = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "Thriller": 53, "War": 10752, "Western": 37,
}
TMDB_ID_TO_NAME = {v: k for k, v in TMDB_GENRE_MAP.items()}

# ─────────────────────────────────────────────
#  TMDB FETCHER (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_tmdb_movies(genres_to_fetch: list[str], pages_per_genre: int = 2) -> pd.DataFrame:
    """Fetch movies from TMDB — cached to avoid repeated calls."""
    rows = []
    seen_ids = set()
    total = len(genres_to_fetch) * pages_per_genre
    done = 0

    progress_text = st.empty()
    progress_bar = st.progress(0, text="🍿 Fetching movies from TMDB...")

    for genre_name in genres_to_fetch:
        genre_id = TMDB_GENRE_MAP.get(genre_name)
        if not genre_id:
            continue
        for page in range(1, pages_per_genre + 1):
            try:
                url = "https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": TMDB_API_KEY,
                    "with_genres": genre_id,
                    "sort_by": "popularity.desc",
                    "page": page,
                    "language": "en-US",
                    "vote_count.gte": 50,
                }
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                for m in data.get("results", []):
                    mid = m.get("id")
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    genre_ids = m.get("genre_ids", [])
                    genres_str = ", ".join(TMDB_ID_TO_NAME.get(gid, "") for gid in genre_ids if gid in TMDB_ID_TO_NAME)
                    poster_path = m.get("poster_path", "")
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
                    year_str = str(m.get("release_date", ""))[:4]
                    try:
                        year = int(year_str) if year_str.isdigit() else None
                    except:
                        year = None
                    rows.append({
                        "tmdb_id": mid,
                        "title": m.get("title", ""),
                        "year": year,
                        "seed_genre": genre_name,
                        "genres": genres_str,
                        "plot": m.get("overview", ""),
                        "rating": round(m.get("vote_average", 0), 1),
                        "popularity": m.get("popularity", 0),
                        "poster_url": poster_url,
                    })
            except Exception:
                pass
            done += 1
            pct = done / total
            progress_bar.progress(pct, text=f"🎬 {genre_name} page {page}/{pages_per_genre}")
            time.sleep(0.1)

    progress_bar.empty()
    progress_text.empty()
    df = pd.DataFrame(rows).drop_duplicates(subset=["tmdb_id"]) if rows else pd.DataFrame()
    # drop rows with missing year to enable filtering
    if not df.empty:
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def search_tmdb_by_name(query: str) -> list[dict]:
    """Search TMDB by movie title."""
    if not query or len(query.strip()) < 2:
        return []
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": query.strip(),
            "language": "en-US",
            "page": 1,
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        results = []
        for m in data.get("results", [])[:8]:
            genre_ids = m.get("genre_ids", [])
            genres_str = ", ".join(TMDB_ID_TO_NAME.get(gid, "") for gid in genre_ids if gid in TMDB_ID_TO_NAME)
            poster_path = m.get("poster_path", "")
            year_str = str(m.get("release_date", ""))[:4]
            try:
                year = int(year_str) if year_str.isdigit() else None
            except:
                year = None
            results.append({
                "tmdb_id": m.get("id"),
                "title": m.get("title", ""),
                "year": year,
                "genres": genres_str,
                "plot": m.get("overview", ""),
                "rating": round(m.get("vote_average", 0), 1),
                "popularity": m.get("popularity", 0),
                "poster_url": f"https://image.tmdb.org/t/p/w92{poster_path}" if poster_path else "",
                "poster_url_full": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "",
                "seed_genre": genres_str.split(",")[0].strip() if genres_str else "Unknown",
            })
        return results
    except Exception:
        return []


# ─────────────────────────────────────────────
#  RECOMMENDER ENGINE (with vectorizer persistence)
# ─────────────────────────────────────────────
def build_engine(df: pd.DataFrame):
    """Create TF-IDF matrix and similarity, store vectorizer."""
    df["features"] = (
        df["genres"].fillna("") + " " +
        df["seed_genre"].fillna("") + " " +
        df["plot"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = tfidf.fit_transform(df["features"])
    sim = cosine_similarity(matrix)
    return sim, tfidf, matrix


def get_recommendations(
    df: pd.DataFrame,
    sim,
    tfidf_vec,
    feature_matrix,
    liked_titles: list[str],
    preferred_genres: list[str],
    query_text: str = "",
    min_year: int = 1900,
    max_year: int = 2030,
    min_rating: float = 0.0,
    top_n: int = 10,
) -> pd.DataFrame:
    # Filter by year & rating first
    filtered = df.copy()
    if "year" in filtered.columns:
        filtered = filtered[(filtered["year"] >= min_year) & (filtered["year"] <= max_year)]
    filtered = filtered[filtered["rating"] >= min_rating]

    if filtered.empty:
        return pd.DataFrame()

    # Case 1: free‑text query (no liked movies)
    if not liked_titles and query_text.strip() and not preferred_genres:
        query_vec = tfidf_vec.transform([query_text])
        sim_scores = cosine_similarity(query_vec, feature_matrix[filtered.index]).flatten()
        filtered["_sim_score"] = sim_scores
        filtered = filtered[filtered["_sim_score"] > 0]
        if filtered.empty:
            return pd.DataFrame()
        filtered["match_score"] = filtered["_sim_score"]
        return filtered.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)

    # Case 2: liked movies (with optional genre preference & text)
    title_lower = filtered["title"].str.lower()
    liked_indices = []
    for t in liked_titles:
        matches = filtered[title_lower == t.lower()]
        if not matches.empty:
            liked_indices.append(matches.index[0])

    if liked_indices:
        agg_scores = None
        for idx in liked_indices:
            # map idx to original position in feature_matrix (index)
            original_pos = filtered.index.get_loc(idx)
            s = sim[original_pos]
            agg_scores = s if agg_scores is None else agg_scores + s
        agg_scores = agg_scores / len(liked_indices)
        filtered["_sim_score"] = agg_scores
        # exclude liked movies themselves
        filtered = filtered[~filtered["title"].str.lower().isin([t.lower() for t in liked_titles])]
    else:
        # no liked movies, use genre preference only
        filtered["_sim_score"] = 0.0

    # genre boost
    if preferred_genres:
        def genre_boost(row):
            g = (row["genres"] or "").lower()
            hits = sum(1 for pg in preferred_genres if pg.lower() in g)
            return hits * 0.05
        filtered["_genre_boost"] = filtered.apply(genre_boost, axis=1)
    else:
        filtered["_genre_boost"] = 0.0

    # if we have both liked and query_text, mix similarities
    if liked_indices and query_text.strip():
        query_vec = tfidf_vec.transform([query_text])
        query_sim = cosine_similarity(query_vec, feature_matrix[filtered.index]).flatten()
        filtered["_text_score"] = query_sim
        filtered["match_score"] = (filtered["_sim_score"] * 0.7) + (filtered["_text_score"] * 0.3) + filtered["_genre_boost"]
    else:
        filtered["match_score"] = filtered["_sim_score"] + filtered["_genre_boost"]

    # fallback: if score is zero (no liked and no genre boost) -> use rating
    if filtered["match_score"].sum() == 0:
        filtered["match_score"] = filtered["rating"] / 10
    return filtered.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
#  CARD RENDERER (same as before)
# ─────────────────────────────────────────────
def render_movie_card(row: pd.Series, rank: int, show_score: bool = True):
    title = row.get("title", "Unknown")
    year = row.get("year", "")
    genres_str = row.get("genres", "")
    plot = row.get("plot", "")
    rating = row.get("rating", "")
    score = row.get("match_score", None)
    poster_url = row.get("poster_url", "")

    poster_html = (
        f'<img class="card-poster" src="{poster_url}" alt="{title}" loading="lazy"/>'
        if poster_url else
        '<div class="card-poster-placeholder">🎬</div>'
    )

    genre_pills = ""
    if genres_str:
        for g in str(genres_str).split(",")[:3]:
            g = g.strip()
            if g:
                genre_pills += f'<span class="card-genre-pill">{g}</span>'

    rating_html = ""
    if rating:
        try:
            r = float(rating)
            stars = "⭐" * min(int(r / 2), 5)
            rating_html = f'<div class="card-rating">{stars} {r:.1f}/10</div>'
        except Exception:
            pass

    score_badge = ""
    if show_score and score is not None:
        pct = min(int(score * 100), 99)
        score_badge = f'<div class="card-score">🎯 {pct}%</div>'

    st.markdown(f"""
    <div class="movie-card">
        {poster_html}
        <div class="card-rank">{rank}</div>
        {score_badge}
        <div class="card-body">
            <div class="card-title">{title}</div>
            <div class="card-meta">{year}</div>
            <div style="margin-bottom:0.3rem">{genre_pills}</div>
            <div class="card-plot">{plot}</div>
            {rating_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "sim" not in st.session_state:
    st.session_state.sim = None
if "tfidf_vec" not in st.session_state:
    st.session_state.tfidf_vec = None
if "feature_matrix" not in st.session_state:
    st.session_state.feature_matrix = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "added_from_search" not in st.session_state:
    st.session_state.added_from_search = []

# Auto‑load on first run (default genres)
DEFAULT_GENRES = ["Action", "Drama", "Science Fiction", "Thriller", "Comedy"]
if st.session_state.df is None:
    with st.spinner("🍿 Auto‑loading movies for you (default genres)... one sec!"):
        df = fetch_tmdb_movies(DEFAULT_GENRES, pages_per_genre=2)
        if not df.empty:
            sim, vec, mat = build_engine(df)
            st.session_state.df = df
            st.session_state.sim = sim
            st.session_state.tfidf_vec = vec
            st.session_state.feature_matrix = mat
            st.session_state.scrape_done = True
        else:
            st.warning("Couldn't load initial movies. Check your internet or TMDB API key.")

# ─────────────────────────────────────────────
#  SIDEBAR (collapsible by Streamlit natively)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🍿 CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">ur personal movie bestie (no cap)</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="input-label">🎭 Pick ur genres</div>', unsafe_allow_html=True)
    scrape_genres = st.multiselect(
        "Genres",
        options=ALL_GENRES,
        default=DEFAULT_GENRES,
        label_visibility="collapsed",
    )
    pages = st.slider("Pages per genre (20 movies/page)", 1, 5, 2,
                      help="More pages = more movies to match against.")
    if st.button("🔄 Refresh Data"):
        if not scrape_genres:
            st.warning("Pick at least one genre 😭")
        else:
            with st.spinner("Fetching fresh movies..."):
                df = fetch_tmdb_movies(scrape_genres, pages)
                if df.empty:
                    st.error("No movies found. Try different genres.")
                else:
                    sim, vec, mat = build_engine(df)
                    st.session_state.df = df
                    st.session_state.sim = sim
                    st.session_state.tfidf_vec = vec
                    st.session_state.feature_matrix = mat
                    st.session_state.recommendations = None
                    st.success(f"✅ Loaded {len(df):,} movies!")
                    st.rerun()

    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"📦 **{len(df_info):,} movies** active<br>⭐ Avg rating: {df_info['rating'].mean():.1f}/10", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*Powered by TMDB API*")

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">🍿 CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">tell us what u like. we find what u watch next. no cap.</div>', unsafe_allow_html=True)
st.markdown("---")

if st.session_state.df is None:
    st.markdown("""
    <div class="empty-state">
        <div class="emoji">😭</div>
        <h3>No movies loaded</h3>
        <p>Use the sidebar to pick genres and click<br><b>🔄 Refresh Data</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df
sim = st.session_state.sim
tfidf_vec = st.session_state.tfidf_vec
feature_matrix = st.session_state.feature_matrix
all_titles = sorted(df["title"].unique().tolist())

tab_rec, tab_search = st.tabs(["🎯 Get Recommendations", "🔍 Search by Name"])

# ══════════════════════════════════════════════
#  TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab_rec:
    st.markdown('<div class="section-header">🎯 Your Vibe <span class="pill">tell us</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        liked_movies = st.multiselect("🍿 movies u already loved", options=all_titles, placeholder="type a title...")
        # also include any movies added from search tab
        if st.session_state.added_from_search:
            added_titles = [m["title"] for m in st.session_state.added_from_search if m["title"] in all_titles]
            for t in added_titles:
                if t not in liked_movies:
                    liked_movies.append(t)
        freetext = st.text_area("📝 or describe the vibe", placeholder="e.g. mind-bending sci-fi with a twist ending...", height=90)
    with col2:
        preferred_genres = st.multiselect("🎭 genres u fw", options=ALL_GENRES, placeholder="pick genres...")
        top_n = st.slider("🔢 how many recs?", 5, 20, 10)

    with st.expander("🎚️ Advanced filters (year / rating)"):
        min_year, max_year = st.slider("Release year", 1900, 2025, (1990, 2025))
        min_rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 5.0, 0.5)

    if st.button("✨ Give Me Recs!"):
        if not liked_movies and not preferred_genres and not freetext.strip():
            st.warning("😭 pick at least one movie, genre, or vibe")
        else:
            with st.spinner("cooking up recommendations... 🍳"):
                recs = get_recommendations(
                    df, sim, tfidf_vec, feature_matrix,
                    liked_movies, preferred_genres, freetext,
                    min_year, max_year, min_rating, top_n
                )
            st.session_state.recommendations = recs

    if st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        if recs.empty:
            st.warning("No movies match your filters. Try broader year/rating or different tastes.")
        else:
            st.markdown(f'<div class="section-header">🎬 Your Recs <span class="pill">top {len(recs)}</span></div>', unsafe_allow_html=True)
            tags_html = "".join(f'<span class="tag">🎬 {m}</span>' for m in liked_movies[:5])
            tags_html += "".join(f'<span class="tag">🎭 {g}</span>' for g in preferred_genres[:5])
            if tags_html:
                st.markdown(f"<div style='margin-bottom:1.2rem'>{tags_html}</div>", unsafe_allow_html=True)

            cols = st.columns(5, gap="medium")
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    render_movie_card(row, rank=i+1)

# ══════════════════════════════════════════════
#  TAB 2 — SEARCH BY NAME (+ fetch similar)
# ══════════════════════════════════════════════
with tab_search:
    st.markdown('<div class="section-header">🔍 Search <span class="pill">by name</span></div>', unsafe_allow_html=True)

    search_query = st.text_input("Movie title", placeholder="e.g. Inception", label_visibility="collapsed")
    if st.button("🔍 Search!"):
        if search_query.strip():
            with st.spinner("Searching TMDB..."):
                results = search_tmdb_by_name(search_query)
            st.session_state.search_results = results
        else:
            st.warning("Enter a movie name")

    if "search_results" in st.session_state and st.session_state.search_results:
        results = st.session_state.search_results
        st.markdown(f"**Found {len(results)} results**")
        for i, m in enumerate(results):
            with st.container():
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    if m.get("poster_url"):
                        st.image(m["poster_url"], width=80)
                    else:
                        st.markdown("🎬")
                with col_b:
                    st.markdown(f"**{m['title']}** ({m.get('year','?')})  ⭐ {m.get('rating','?')}/10")
                    st.markdown(f"*{', '.join(m['genres'].split(',')[:3]) if m['genres'] else 'No genres'}*")
                    st.caption(m.get("plot", "")[:200])
                    # Two buttons: View details & Find similar
                    if st.button(f"📖 Details & Similar", key=f"detail_{m['tmdb_id']}_{i}"):
                        st.session_state.selected_movie = m
        st.markdown("---")

    # Show detailed view of selected movie + similar movies button
    if "selected_movie" in st.session_state:
        sel = st.session_state.selected_movie
        st.markdown("### 🎬 Movie Details")
        d1, d2 = st.columns([1, 2])
        with d1:
            if sel.get("poster_url_full"):
                st.image(sel["poster_url_full"], use_container_width=True)
            else:
                st.markdown("🎬 *No poster*")
        with d2:
            st.markdown(f"## {sel['title']}")
            st.markdown(f"**Year:** {sel.get('year','?')}  |  **Rating:** {sel.get('rating','?')}/10")
            st.markdown(f"**Genres:** {sel.get('genres','')}")
            st.markdown(f"**Overview:** {sel.get('plot','No plot available')}")

        if st.button("✨ Find similar movies (based on this movie's plot & genres)"):
            # Use the movie's plot+genres as a query to get recs from the loaded dataset
            query_text = f"{sel.get('genres','')} {sel.get('plot','')}"
            sim_recs = get_recommendations(
                df, sim, tfidf_vec, feature_matrix,
                liked_titles=[],  # no liked movies, just this description
                preferred_genres=[],
                query_text=query_text,
                min_year=1900,
                max_year=2025,
                min_rating=0,
                top_n=10
            )
            if not sim_recs.empty:
                st.markdown("#### Movies like this:")
                cols = st.columns(5)
                for idx, (_, row) in enumerate(sim_recs.iterrows()):
                    with cols[idx % 5]:
                        render_movie_card(row, rank=idx+1, show_score=True)
                # Offer to add the original movie to liked list for the other tab
                if sel["title"] in all_titles:
                    if st.button(f"➕ Add '{sel['title']}' to my liked movies (for future recs)"):
                        if sel not in st.session_state.added_from_search:
                            st.session_state.added_from_search.append(sel)
                            st.success(f"Added! Go to 🎯 Get Recommendations tab ✨")
                else:
                    st.info("💡 This movie isn't in your current dataset. Load more genres/pages to see it in recommendations.")
            else:
                st.warning("No similar movies found. Try broadening your dataset (more genres/pages).")

        if st.button("Clear selection"):
            del st.session_state.selected_movie
            st.rerun()
