import streamlit as st
import pandas as pd
import requests
import json
import time
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0c0c10;
    --surface:   #13131a;
    --surface2:  #1c1c27;
    --accent:    #e8b84b;
    --accent2:   #d4507a;
    --text:      #f0eee8;
    --muted:     #7a7893;
    --border:    rgba(232,184,75,0.15);
    --radius:    14px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Hero Title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #e8b84b 0%, #f5d98e 45%, #d4507a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* ── Movie Card ── */
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
    position: relative;
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: rgba(232,184,75,0.45);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5);
}

.card-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}

.card-poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: linear-gradient(135deg, var(--surface2) 0%, #1a1a28 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
}

.card-body {
    padding: 1rem;
}

.card-rank {
    position: absolute;
    top: 10px;
    left: 10px;
    background: var(--accent);
    color: #0c0c10;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 0.85rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.card-score {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(12,12,16,0.85);
    backdrop-filter: blur(6px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
}

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.25rem;
    line-height: 1.3;
}

.card-meta {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

.card-genre-pill {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--muted);
    margin-right: 4px;
    margin-bottom: 4px;
}

.card-plot {
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.card-rating {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.82rem;
    color: var(--accent);
    font-weight: 500;
    margin-top: 0.5rem;
}

/* ── Section Header ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.section-header span {
    color: var(--accent);
}

/* ── Input section ── */
.input-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* ── Sidebar ── */
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.2rem;
}

.sidebar-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

/* ── Tag pill ── */
.tag {
    display: inline-block;
    background: rgba(232,184,75,0.12);
    border: 1px solid rgba(232,184,75,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin-right: 5px;
    margin-bottom: 5px;
}

/* ── Override streamlit widgets ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stMultiSelect"] > div,
div[data-testid="stSelectbox"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stSlider [data-testid="stSlider"] { accent-color: var(--accent); }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent) 0%, #d4a03a 100%) !important;
    color: #0c0c10 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s ease !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Alert/info boxes ── */
div[data-testid="stAlert"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── No results ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}

.empty-state .emoji { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 { font-family: 'Playfair Display', serif; color: var(--text); }

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
OMDB_API_KEY = "trilogy"   # free key — limited; swap with your own

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "History", "Horror", "Music", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

IMDB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────────────────────────
#  SCRAPER
# ─────────────────────────────────────────────
def scrape_imdb(genres_to_scrape: list[str], pages_per_genre: int = 2) -> pd.DataFrame:
    """Scrape IMDB search results for given genres."""
    rows = []
    seen = set()

    total = len(genres_to_scrape) * pages_per_genre
    done = 0

    bar = st.progress(0, text="🎬 Fetching movies from IMDB...")

    for genre in genres_to_scrape:
        for page in range(1, pages_per_genre + 1):
            start = (page - 1) * 50 + 1
            url = (
                f"https://www.imdb.com/search/title/"
                f"?genres={genre.lower()}&start={start}&count=50"
            )
            try:
                r = requests.get(url, headers=IMDB_HEADERS, timeout=15)
                soup = BeautifulSoup(r.text, "html.parser")
                movies = soup.select(".lister-item")

                for m in movies:
                    try:
                        title = m.h3.a.text.strip()
                        if title in seen:
                            continue
                        seen.add(title)

                        year = m.select_one(".lister-item-year")
                        year = year.get_text(strip=True) if year else ""

                        rating_tag = m.select_one(".ratings-imdb-rating")
                        rating = rating_tag["data-value"] if rating_tag else None

                        genre_tag = m.select_one(".genre")
                        genres_found = genre_tag.text.strip() if genre_tag else ""

                        plot_tags = m.select(".text-muted")
                        plot = plot_tags[2].text.strip() if len(plot_tags) >= 3 else ""

                        rows.append({
                            "title": title,
                            "seed_genre": genre,
                            "genres": genres_found,
                            "plot": plot,
                            "rating": rating,
                            "year": year,
                        })
                    except Exception:
                        continue

            except Exception:
                pass

            done += 1
            pct = done / total
            bar.progress(pct, text=f"🎬 Scraping {genre} — page {page}/{pages_per_genre}")
            time.sleep(0.8)

    bar.empty()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  POSTER FETCHER (OMDb)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_poster_url(title: str, year: str = "") -> str | None:
    """Fetch poster URL from OMDb API."""
    clean_year = year.replace("(", "").replace(")", "").strip()[:4]
    params = {"t": title, "apikey": OMDB_API_KEY}
    if clean_year.isdigit():
        params["y"] = clean_year
    try:
        r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
        data = r.json()
        poster = data.get("Poster", "")
        return poster if poster and poster != "N/A" else None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  RECOMMENDER ENGINE
# ─────────────────────────────────────────────
def build_engine(df: pd.DataFrame):
    """Build TF-IDF cosine similarity matrix."""
    df["features"] = (
        df["genres"].fillna("") + " " +
        df["seed_genre"].fillna("") + " " +
        df["plot"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = tfidf.fit_transform(df["features"])
    sim = cosine_similarity(matrix)
    return sim


def get_recommendations(
    df: pd.DataFrame,
    sim: any,
    liked_titles: list[str],
    preferred_genres: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N recommendations based on liked titles + preferred genres."""

    # Build a genre boost vector
    genre_str = " ".join(preferred_genres).lower()

    # Find indices of liked movies
    title_lower = df["title"].str.lower()
    liked_indices = []
    for t in liked_titles:
        matches = df[title_lower == t.lower()]
        if not matches.empty:
            liked_indices.append(matches.index[0])

    # ── Fallback: use genre filtering if no liked titles found ──
    if not liked_indices:
        mask = df["genres"].fillna("").str.lower()
        for g in preferred_genres:
            mask = mask.str.contains(g.lower(), na=False)
        subset = df[mask].copy()
        if subset.empty:
            subset = df.copy()
        subset["score"] = pd.to_numeric(subset["rating"], errors="coerce").fillna(0)
        return subset.sort_values("score", ascending=False).head(top_n)

    # ── Aggregate similarity scores ──
    agg_scores = None
    for idx in liked_indices:
        s = sim[idx]
        agg_scores = s if agg_scores is None else agg_scores + s

    agg_scores = agg_scores / len(liked_indices)

    # Exclude already-liked titles
    result = df.copy()
    result["_sim_score"] = agg_scores
    liked_set = {t.lower() for t in liked_titles}
    result = result[~result["title"].str.lower().isin(liked_set)]

    # Genre boost
    if preferred_genres:
        def genre_boost(row):
            g = (row["genres"] or "").lower()
            hits = sum(1 for pg in preferred_genres if pg.lower() in g)
            return hits * 0.05
        result["_genre_boost"] = result.apply(genre_boost, axis=1)
    else:
        result["_genre_boost"] = 0

    result["_final_score"] = result["_sim_score"] + result["_genre_boost"]
    result = result.sort_values("_final_score", ascending=False).head(top_n)
    result = result.rename(columns={"_final_score": "match_score"})
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────
#  MOVIE CARD HTML
# ─────────────────────────────────────────────
def render_movie_card(row: pd.Series, rank: int, show_score: bool = True):
    title = row.get("title", "Unknown")
    year = row.get("year", "")
    genres_str = row.get("genres", "")
    plot = row.get("plot", "")
    rating = row.get("rating", "")
    score = row.get("match_score", None)

    poster_url = get_poster_url(title, year)

    # Poster HTML
    if poster_url:
        poster_html = f'<img class="card-poster" src="{poster_url}" alt="{title}" loading="lazy"/>'
    else:
        poster_html = '<div class="card-poster-placeholder">🎬</div>'

    # Genre pills
    genre_pills = ""
    if genres_str:
        for g in genres_str.split(",")[:3]:
            g = g.strip()
            if g:
                genre_pills += f'<span class="card-genre-pill">{g}</span>'

    # Rating stars
    rating_html = ""
    if rating:
        try:
            r = float(rating)
            rating_html = f'<div class="card-rating">⭐ {r:.1f} / 10</div>'
        except Exception:
            pass

    # Score badge
    score_badge = ""
    if show_score and score is not None:
        pct = min(int(score * 100), 99)
        score_badge = f'<div class="card-score">{pct}% match</div>'

    st.markdown(f"""
    <div class="movie-card">
        {poster_html}
        <div class="card-rank">{rank}</div>
        {score_badge}
        <div class="card-body">
            <div class="card-title">{title}</div>
            <div class="card-meta">{year}</div>
            <div style="margin-bottom:0.5rem">{genre_pills}</div>
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
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "scrape_done" not in st.session_state:
    st.session_state.scrape_done = False


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Content-Based Movie Recommender</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="input-label">⚙️ Data Source</div>', unsafe_allow_html=True)

    scrape_genres = st.multiselect(
        "Genres to fetch from IMDB",
        options=ALL_GENRES,
        default=["Action", "Drama", "Sci-Fi", "Thriller", "Comedy"],
        help="Select which genres to scrape from IMDB. More genres = more data = better results.",
    )

    pages = st.slider(
        "Pages per genre (50 movies/page)",
        min_value=1, max_value=5, value=2,
        help="More pages = more movies = slower scraping."
    )

    st.markdown(
        f"<div style='font-size:0.78rem;color:var(--muted);margin-top:0.3rem;'>"
        f"~{len(scrape_genres) * pages * 50} movies estimated</div>",
        unsafe_allow_html=True
    )

    if st.button("🚀 Fetch Movies from IMDB"):
        if not scrape_genres:
            st.warning("Pick at least one genre.")
        else:
            with st.spinner("Connecting to IMDB..."):
                df = scrape_imdb(scrape_genres, pages)
            if df.empty:
                st.error("No data returned. Try again or check your connection.")
            else:
                sim = build_engine(df)
                st.session_state.df = df
                st.session_state.sim = sim
                st.session_state.scrape_done = True
                st.session_state.recommendations = None
                st.success(f"✅ {len(df):,} movies loaded!")

    st.markdown("---")

    if st.session_state.scrape_done and st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(
            f"<div style='font-size:0.82rem;color:var(--muted);'>"
            f"📦 Dataset: <b style='color:var(--accent)'>{len(df):,}</b> unique movies<br>"
            f"🎭 Genres: <b style='color:var(--accent)'>{df['seed_genre'].nunique()}</b> categories"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:var(--muted);line-height:1.6;'>"
        "Built with Cosine Similarity on TF-IDF vectors<br>"
        "Features: Genre tags + Plot synopsis<br>"
        "Posters via OMDb API"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">CineMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Tell us what you love — we find what you\'ll watch next.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── No data yet ──
if not st.session_state.scrape_done:
    st.markdown("""
    <div class="empty-state">
        <div class="emoji">🍿</div>
        <h3>Start by fetching movies</h3>
        <p>Use the sidebar to select genres and hit <b>Fetch Movies from IMDB</b>.<br>
        Then come back here to get personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Preference Form ──
df = st.session_state.df
sim = st.session_state.sim
all_titles = sorted(df["title"].unique().tolist())

st.markdown('<div class="section-header">🎯 Your <span>Preferences</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="input-label">🎬 Movies You\'ve Loved</div>', unsafe_allow_html=True)
    liked_movies = st.multiselect(
        "",
        options=all_titles,
        placeholder="Start typing a movie title...",
        help="Select movies you've already watched and enjoyed. The more you pick, the better.",
        label_visibility="collapsed",
    )

    st.markdown('<div class="input-label" style="margin-top:1rem">📝 Or describe what you want</div>', unsafe_allow_html=True)
    freetext = st.text_area(
        "",
        placeholder="e.g. mind-bending sci-fi with a twist ending, or a war epic with emotional depth...",
        height=100,
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<div class="input-label">🎭 Preferred Genres</div>', unsafe_allow_html=True)
    preferred_genres = st.multiselect(
        "",
        options=ALL_GENRES,
        placeholder="Select genres you like...",
        label_visibility="collapsed",
    )

    st.markdown('<div class="input-label" style="margin-top:1rem">🔢 How many recommendations?</div>', unsafe_allow_html=True)
    top_n = st.slider("", min_value=5, max_value=20, value=10, label_visibility="collapsed")

    st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
    go_btn = st.button("✨ Find My Movies")


# ── Run Recommender ──
if go_btn:
    if not liked_movies and not preferred_genres and not freetext.strip():
        st.warning("⚠️ Please select at least one movie or genre to get recommendations.")
        st.stop()

    # If freetext, add it to liked titles for matching (fuzzy approach)
    lookup_titles = liked_movies[:]

    with st.spinner("Computing cosine similarities..."):
        recs = get_recommendations(df, sim, lookup_titles, preferred_genres, top_n)

    st.session_state.recommendations = recs


# ── Display Results ──
if st.session_state.recommendations is not None:
    recs = st.session_state.recommendations

    st.markdown(
        f'<div class="section-header">🎬 Your <span>Recommendations</span>'
        f'<span style="font-size:1rem;color:var(--muted);font-family:\'DM Sans\',sans-serif;font-weight:400;"> — Top {len(recs)}</span></div>',
        unsafe_allow_html=True,
    )

    # Input tags display
    if liked_movies or preferred_genres:
        tags_html = ""
        for m in liked_movies:
            tags_html += f'<span class="tag">🎬 {m}</span>'
        for g in preferred_genres:
            tags_html += f'<span class="tag">🎭 {g}</span>'
        st.markdown(f"<div style='margin-bottom:1.5rem'>{tags_html}</div>", unsafe_allow_html=True)

    # Grid — 5 columns
    cols = st.columns(5, gap="medium")
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            render_movie_card(row, rank=i + 1)

    # Stats expander
    with st.expander("📊 Dataset Stats", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Movies", f"{len(df):,}")
        with c2:
            avg_r = pd.to_numeric(df["rating"], errors="coerce").mean()
            st.metric("Avg IMDB Rating", f"{avg_r:.2f}")
        with c3:
            st.metric("Genres Covered", df["seed_genre"].nunique())

        genre_counts = df["seed_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        st.dataframe(genre_counts, use_container_width=True, hide_index=True)
