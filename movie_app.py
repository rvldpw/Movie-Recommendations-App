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
#  CUSTOM CSS  —  FUN / MEME THEME
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

/* Sidebar */
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

/* Section headers */
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
div[data-testid="stTextInput"] input {
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

/* Tag pills */
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

/* Empty state */
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
.empty-state p { color: #555; font-weight: 600; }

/* Sidebar label */
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

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'Fredoka One', cursive !important;
    font-size: 1rem !important;
    color: var(--dark) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid var(--pink) !important;
    color: var(--pink) !important;
}

/* Alert */
div[data-testid="stAlert"] {
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
}

hr { border-color: var(--border) !important; border-width: 2px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
# Get a free TMDB API key at: https://www.themoviedb.org/settings/api
# Works much better than IMDB scraping!
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # public demo key

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
#  TMDB FETCHER
# ─────────────────────────────────────────────
def fetch_tmdb_movies(genres_to_fetch: list[str], pages_per_genre: int = 2) -> pd.DataFrame:
    """Fetch movies from TMDB API — reliable, fast, no blocking."""
    rows = []
    seen_ids = set()

    total = len(genres_to_fetch) * pages_per_genre
    done = 0
    bar = st.progress(0, text="🍿 Fetching movies from TMDB...")

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

                    rows.append({
                        "tmdb_id": mid,
                        "title": m.get("title", ""),
                        "year": str(m.get("release_date", ""))[:4],
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
            bar.progress(pct, text=f"🎬 Fetching {genre_name} — page {page}/{pages_per_genre}")
            time.sleep(0.2)

    bar.empty()
    return pd.DataFrame(rows).drop_duplicates(subset=["tmdb_id"]) if rows else pd.DataFrame()


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
            results.append({
                "tmdb_id": m.get("id"),
                "title": m.get("title", ""),
                "year": str(m.get("release_date", ""))[:4],
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
#  RECOMMENDER ENGINE
# ─────────────────────────────────────────────
def build_engine(df: pd.DataFrame):
    df["features"] = (
        df["genres"].fillna("") + " " +
        df["seed_genre"].fillna("") + " " +
        df["plot"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = tfidf.fit_transform(df["features"])
    return cosine_similarity(matrix)


def get_recommendations(
    df: pd.DataFrame,
    sim,
    liked_titles: list[str],
    preferred_genres: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    title_lower = df["title"].str.lower()
    liked_indices = []
    for t in liked_titles:
        matches = df[title_lower == t.lower()]
        if not matches.empty:
            liked_indices.append(matches.index[0])

    if not liked_indices:
        mask = pd.Series([True] * len(df), index=df.index)
        if preferred_genres:
            mask = df["genres"].fillna("").apply(
                lambda g: any(pg.lower() in g.lower() for pg in preferred_genres)
            )
        subset = df[mask].copy()
        if subset.empty:
            subset = df.copy()
        subset["match_score"] = pd.to_numeric(subset["rating"], errors="coerce").fillna(0) / 10
        return subset.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)

    agg_scores = None
    for idx in liked_indices:
        s = sim[idx]
        agg_scores = s if agg_scores is None else agg_scores + s
    agg_scores = agg_scores / len(liked_indices)

    result = df.copy()
    result["_sim_score"] = agg_scores
    liked_set = {t.lower() for t in liked_titles}
    result = result[~result["title"].str.lower().isin(liked_set)]

    if preferred_genres:
        def genre_boost(row):
            g = (row["genres"] or "").lower()
            hits = sum(1 for pg in preferred_genres if pg.lower() in g)
            return hits * 0.05
        result["_genre_boost"] = result.apply(genre_boost, axis=1)
    else:
        result["_genre_boost"] = 0

    result["match_score"] = result["_sim_score"] + result["_genre_boost"]
    return result.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
#  CARD RENDERER
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
#  SESSION STATE
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "sim" not in st.session_state:
    st.session_state.sim = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "scrape_done" not in st.session_state:
    st.session_state.scrape_done = False
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "added_from_search" not in st.session_state:
    st.session_state.added_from_search = []  # list of dicts from TMDB search
if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🍿 CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">ur personal movie bestie (no cap)</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="input-label">🎭 Pick ur genres</div>', unsafe_allow_html=True)

    scrape_genres = st.multiselect(
        "Genres",
        options=ALL_GENRES,
        default=["Action", "Drama", "Science Fiction", "Thriller", "Comedy"],
        label_visibility="collapsed",
    )

    pages = st.slider(
        "Pages per genre (20 movies/page)",
        min_value=1, max_value=5, value=2,
        help="More pages = more movies to match against."
    )

    est = len(scrape_genres) * pages * 20
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:700;color:#555;margin-top:0.2rem;'>"
        f"~{est} movies estimated 🎬</div>",
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin-top:0.8rem"></div>', unsafe_allow_html=True)

    if st.button("🚀 Load Movies!"):
        if not scrape_genres:
            st.warning("Pick at least one genre bestie 😭")
        else:
            with st.spinner("grabbing movies from TMDB... one sec 🍿"):
                df = fetch_tmdb_movies(scrape_genres, pages)
            if df.empty:
                st.error("😩 Nothing came back. Check your internet or try fewer genres.")
            else:
                sim = build_engine(df)
                st.session_state.df = df
                st.session_state.sim = sim
                st.session_state.scrape_done = True
                st.session_state.recommendations = None
                st.success(f"✅ {len(df):,} movies loaded! slay 💅")

    st.markdown("---")

    if st.session_state.scrape_done and st.session_state.df is not None:
        df_s = st.session_state.df
        st.markdown(
            f"<div style='font-size:0.82rem;font-weight:700;'>"
            f"📦 <b>{len(df_s):,}</b> movies loaded<br>"
            f"🎭 <b>{df_s['seed_genre'].nunique()}</b> genre categories<br>"
            f"⭐ Avg rating: <b>{pd.to_numeric(df_s['rating'], errors='coerce').mean():.1f}</b>/10"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#666;line-height:1.8;'>"
        "Powered by TMDB API 🎬<br>"
        "TF-IDF + Cosine Similarity<br>"
        "Built different 😤"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">🍿 CineMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">tell us what u like. we find what u watch next. no cap.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── No data yet ──
if not st.session_state.scrape_done:
    st.markdown("""
    <div class="empty-state">
        <div class="emoji">👈</div>
        <h3>load movies first bestie</h3>
        <p>Pick genres in the sidebar and hit<br><b>🚀 Load Movies!</b><br>then come back here for recs ✨</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  TABS: Recommend  |  Search by Name
# ─────────────────────────────────────────────
df = st.session_state.df
sim = st.session_state.sim
all_titles = sorted(df["title"].unique().tolist())

tab_rec, tab_search = st.tabs(["🎯 Get Recommendations", "🔍 Search by Name"])


# ══════════════════════════════════════════════
#  TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab_rec:
    st.markdown(
        '<div class="section-header">🎯 Your Vibe <span class="pill">tell us</span></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-label">🎬 movies u already loved</div>', unsafe_allow_html=True)
        liked_movies = st.multiselect(
            "liked",
            options=all_titles,
            placeholder="type a title...",
            label_visibility="collapsed",
        )

        # Also allow adding from name-search tab
        if st.session_state.added_from_search:
            added_titles = [m["title"] for m in st.session_state.added_from_search]
            # Merge with liked_movies (dedup)
            for t in added_titles:
                if t not in liked_movies and t in all_titles:
                    liked_movies.append(t)
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;color:#888;margin-top:4px;'>"
                + " ".join(f"<span class='tag'>🔍 {t}</span>" for t in added_titles)
                + "</div>",
                unsafe_allow_html=True
            )

        st.markdown('<div class="input-label" style="margin-top:1rem">📝 or describe the vibe</div>', unsafe_allow_html=True)
        freetext = st.text_area(
            "vibe",
            placeholder="e.g. mind-bending sci-fi with a twist ending... or a movie that made u cry lol",
            height=90,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown('<div class="input-label">🎭 genres u fw</div>', unsafe_allow_html=True)
        preferred_genres = st.multiselect(
            "genres",
            options=ALL_GENRES,
            placeholder="pick genres...",
            label_visibility="collapsed",
        )

        st.markdown('<div class="input-label" style="margin-top:1rem">🔢 how many recs?</div>', unsafe_allow_html=True)
        top_n = st.slider("n", min_value=5, max_value=20, value=10, label_visibility="collapsed")

        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        go_btn = st.button("✨ Give Me Recs!")

    if go_btn:
        if not liked_movies and not preferred_genres and not freetext.strip():
            st.warning("😭 pick at least one movie or genre bestie")
            st.stop()

        with st.spinner("cooking up recommendations... 🍳"):
            recs = get_recommendations(df, sim, liked_movies, preferred_genres, top_n)

        st.session_state.recommendations = recs

    if st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        st.markdown(
            f'<div class="section-header">🎬 Your Recs <span class="pill">top {len(recs)}</span></div>',
            unsafe_allow_html=True,
        )

        tags_html = ""
        for m in liked_movies:
            tags_html += f'<span class="tag">🎬 {m}</span>'
        for g in preferred_genres:
            tags_html += f'<span class="tag">🎭 {g}</span>'
        if tags_html:
            st.markdown(f"<div style='margin-bottom:1.2rem'>{tags_html}</div>", unsafe_allow_html=True)

        cols = st.columns(5, gap="medium")
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                render_movie_card(row, rank=i + 1)


# ══════════════════════════════════════════════
#  TAB 2 — SEARCH BY NAME
# ══════════════════════════════════════════════
with tab_search:
    st.markdown(
        '<div class="section-header">🔍 Search <span class="pill">by name</span></div>',
        unsafe_allow_html=True
    )

    search_col, btn_col = st.columns([4, 1])
    with search_col:
        search_query = st.text_input(
            "search",
            placeholder="type a movie name...",
            label_visibility="collapsed",
            key="search_input",
        )
    with btn_col:
        st.markdown('<div style="margin-top:0.15rem"></div>', unsafe_allow_html=True)
        search_btn = st.button("🔍 Search!")

    if search_btn and search_query.strip():
        with st.spinner("searching TMDB... 🕵️"):
            results = search_tmdb_by_name(search_query)
        st.session_state.search_results = results

    if st.session_state.search_results:
        results = st.session_state.search_results
        st.markdown(
            f"<div style='font-size:0.9rem;font-weight:800;color:#888;margin-bottom:0.8rem;'>"
            f"found {len(results)} results ✅</div>",
            unsafe_allow_html=True
        )

        # Display results as cards (2 columns)
        res_cols = st.columns(2, gap="medium")
        for i, m in enumerate(results):
            with res_cols[i % 2]:
                poster_html = (
                    f'<img class="search-poster" src="{m["poster_url"]}" alt="{m["title"]}"/>'
                    if m.get("poster_url") else
                    '<div class="search-poster-placeholder">🎬</div>'
                )
                genres_pills = "".join(
                    f'<span class="search-genres">{g.strip()}</span>'
                    for g in str(m.get("genres", "")).split(",")[:3] if g.strip()
                )

                st.markdown(f"""
                <div class="search-card">
                    {poster_html}
                    <div style="flex:1;min-width:0;">
                        <div class="search-title">{m["title"]}</div>
                        <div class="search-meta">📅 {m.get("year","?")} &nbsp; ⭐ {m.get("rating","?")}/10</div>
                        <div style="margin-top:4px">{genres_pills}</div>
                        <div style="font-size:0.72rem;color:#555;margin-top:4px;line-height:1.4;
                             display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">
                            {m.get("plot","")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Full movie detail for selected movie
        selected_title = st.selectbox(
            "👀 pick one to see full details",
            options=["—"] + [m["title"] for m in results],
        )

        if selected_title != "—":
            sel = next((m for m in results if m["title"] == selected_title), None)
            if sel:
                d1, d2 = st.columns([1, 3])
                with d1:
                    if sel.get("poster_url_full"):
                        st.image(sel["poster_url_full"], use_container_width=True)
                    else:
                        st.markdown('<div style="font-size:5rem;text-align:center">🎬</div>', unsafe_allow_html=True)
                with d2:
                    st.markdown(
                        f"<div style='font-family:Fredoka One,cursive;font-size:1.8rem;'>{sel['title']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-weight:700;color:#666;margin-bottom:0.5rem;'>"
                        f"📅 {sel.get('year','?')} &nbsp;|&nbsp; ⭐ {sel.get('rating','?')}/10 &nbsp;|&nbsp; 🎭 {sel.get('genres','')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:0.95rem;line-height:1.6;color:#333;'>{sel.get('plot','No plot available.')}</div>",
                        unsafe_allow_html=True
                    )

                    # If this movie is in dataset, offer to add to liked
                    in_dataset = sel["title"] in all_titles
                    if in_dataset:
                        if st.button(f"➕ Add '{sel['title']}' to my liked movies (for recs)"):
                            already = any(m["title"] == sel["title"] for m in st.session_state.added_from_search)
                            if not already:
                                st.session_state.added_from_search.append(sel)
                                st.success(f"Added! Go to 🎯 Get Recommendations tab ✨")
                            else:
                                st.info("Already added!")
                    else:
                        st.info("💡 This movie isn't in your loaded dataset. Load more genres/pages, then it'll show up for recommendations too!")

    elif search_btn:
        st.markdown("""
        <div class="empty-state" style="max-width:300px;padding:2rem">
            <div class="emoji">😶</div>
            <h3>nothing found</h3>
            <p>try a different title?</p>
        </div>
        """, unsafe_allow_html=True)
