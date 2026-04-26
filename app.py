import streamlit as st
import pandas as pd
import requests
import random

from src.data_loader import load_data
from src.recommender import RecommenderSystem

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineWrap 2026",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --red:    #E50914;
    --red2:   #B20710;
    --dark:   #0A0A0A;
    --card:   #141414;
    --card2:  #1C1C1C;
    --border: #2A2A2A;
    --muted:  #808080;
    --white:  #FFFFFF;
    --gold:   #F5C518;
}

.stApp { background: var(--dark) !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #0D0D0D !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #CCC !important; }
[data-testid="collapsedControl"] {
    background: var(--red) !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="collapsedControl"] svg { color: white !important; fill: white !important; }

.stTextInput input {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.2) !important;
}
.stTextInput label { color: #888 !important; font-size:0.8rem !important; }
.stSlider [role="slider"] { background: var(--red) !important; }
.stSlider [data-baseweb="slider-track"] div:first-child { background: var(--red) !important; }
.stToggle label { color: #CCC !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, var(--red2), var(--red)) !important;
    border-radius: 10px !important;
}
.stProgress { background: var(--card2) !important; border-radius: 10px !important; }
[data-testid="stMetricValue"] {
    color: var(--red) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:0.75rem !important; }
hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }
.stAlert { border-radius: 8px !important; }
.stSpinner { color: var(--red) !important; }
h1,h2,h3 { font-family:'Bebas Neue',sans-serif !important; letter-spacing:2px !important; }
h2 { color: var(--white) !important; font-size:1.8rem !important; }
p, li, span { color:#BBB !important; }

/* ── Movie grid ── */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.movie-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.movie-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.movie-grid.cols-5 { grid-template-columns: repeat(5, 1fr); }

/* ── Poster card ── */
.poster-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
    height: 100%;
}
.poster-card:hover {
    border-color: var(--red);
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(229,9,20,0.2);
}
.poster-card .poster-img-wrap {
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background: #1a1a1a;
    flex-shrink: 0;
}
.poster-card .poster-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.poster-info {
    padding: 0.8rem 1rem 1rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.poster-title {
    color: var(--white);
    font-family:'Inter',sans-serif;
    font-weight:600;
    font-size:0.85rem;
    line-height:1.35;
}
.poster-rating { color: var(--gold); font-size:0.78rem; }
.poster-date   { color: var(--muted); font-size:0.73rem; }
.poster-user-rating { color: #BBB; font-size:0.75rem; }
.poster-match  { color: var(--red); font-weight:700; font-size:0.8rem; margin-top: auto; padding-top: 6px; }
.poster-badge  {
    display:inline-block; background:var(--red);
    color:white; font-size:0.7rem; font-weight:700;
    padding:2px 7px; border-radius:4px;
    width: fit-content;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #200000 0%, #0A0A0A 55%);
    border: 1px solid #1E1E1E;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content:'';
    position:absolute; top:0; right:0; width:40%; height:100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(229,9,20,0.15), transparent 70%);
    pointer-events:none;
}
.year-pill {
    display:inline-block; background:var(--red);
    color:white; font-family:'Bebas Neue',sans-serif;
    font-size:0.95rem; letter-spacing:3px;
    padding:3px 14px; border-radius:20px; margin-bottom:0.8rem;
}
.hero-title {
    font-family:'Bebas Neue',sans-serif;
    font-size:3.8rem; color:var(--white);
    letter-spacing:6px; line-height:1; margin:0;
}
.hero-title span { color:var(--red); }
.hero-sub {
    font-family:'Inter',sans-serif; font-size:0.9rem;
    color:var(--muted); margin-top:0.6rem; letter-spacing:2px;
    text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── OMDb ───────────────────────────────────────────────────────────────────────
OMDB_API_KEY = "trilogy"
PLACEHOLDER  = "https://placehold.co/400x600/141414/333333?text=🎬"

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_info(title: str) -> dict:
    clean = title.split("(")[0].strip()
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"t": clean, "apikey": OMDB_API_KEY},
            timeout=4,
        )
        data = r.json()
        if data.get("Response") == "True":
            return {
                "poster": data.get("Poster", ""),
                "imdb":   data.get("imdbRating", "N/A"),
                "year":   data.get("Year", ""),
                "genre":  data.get("Genre", ""),
            }
    except Exception:
        pass
    return {"poster": "", "imdb": "N/A", "year": "", "genre": ""}

def safe_poster(title: str):
    info = fetch_movie_info(title)
    url  = info["poster"] if info["poster"] not in ("", "N/A") else PLACEHOLDER
    return url, info["imdb"]

# ── Username generator ─────────────────────────────────────────────────────────
_USER_ADJ  = ["Silent","Cosmic","Neon","Crimson","Golden","Shadow","Electric",
              "Frozen","Blazing","Midnight","Velvet","Phantom","Stellar","Iron"]
_USER_NOUN = ["Watcher","Cinephile","Director","Critic","Reel","Projector",
              "Auteur","Curator","Maverick","Pioneer","Voyager","Lens","Frame"]

def auto_username(user_id: int) -> str:
    rng = random.Random(int(user_id))
    return f"{rng.choice(_USER_ADJ)}{rng.choice(_USER_NOUN)}{user_id % 100:02d}"

# ── Card HTML helpers ──────────────────────────────────────────────────────────
def recent_card(row, show_posters: bool) -> str:
    img, imdb = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")
    stars = "★" * int(round(row["rating"])) + "☆" * (5 - int(round(row["rating"])))
    return f"""
    <div class="poster-card">
      <div class="poster-img-wrap">
        <img src="{img}" alt="" onerror="this.src='{PLACEHOLDER}'"/>
      </div>
      <div class="poster-info">
        <div class="poster-title">{row['title']}</div>
        <div class="poster-rating">{stars} &nbsp; IMDB {imdb}</div>
        <div class="poster-date">📅 {row['datetime'].date()}</div>
        <div class="poster-user-rating">Your rating: {row['rating']}/5</div>
      </div>
    </div>"""

def rec_card(row, rank: int, show_posters: bool) -> str:
    img, imdb = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")
    pct = int(row["score"] * 100)
    return f"""
    <div class="poster-card">
      <div class="poster-img-wrap">
        <img src="{img}" alt="" onerror="this.src='{PLACEHOLDER}'"/>
      </div>
      <div class="poster-info">
        <div class="poster-badge">#{rank}</div>
        <div class="poster-title">{row['title']}</div>
        <div class="poster-rating">IMDB {imdb}</div>
        <div class="poster-match">🎯 {pct}% match</div>
      </div>
    </div>"""

def render_grid(cards: list[str], cols: int = 5) -> None:
    """Render a list of card HTML strings in a CSS grid."""
    inner = "".join(cards)  # ← Change "\n".join to "".join
    st.markdown(
        f'<div class="movie-grid cols-{cols}">{inner}</div>',
        unsafe_allow_html=True,
    )

# ── Load system ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Starting up CineWrap…")
def load_system() -> RecommenderSystem:
    df    = load_data()
    model = RecommenderSystem(df)
    model.fit()
    return model

system = load_system()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="year-pill">2026 EDITION</div>
  <div class="hero-title">CINE<span>WRAP</span></div>
  <div class="hero-sub">Your taste. Your universe. Mapped.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineWrap 2026")
    st.markdown(
        "<p style='color:#555;font-size:0.78rem;margin-top:-8px;'>Hybrid Movie Recommender</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### 🎯 Recommendations")
    top_n_recs = st.slider("How many picks?", 5, 20, 10)
    st.markdown("#### 📅 Watch History")
    top_n_recent = st.slider("Recent movies shown", 3, 10, 5)
    st.markdown("#### 🖼️ Movie Artwork")
    show_posters = st.toggle("Fetch posters from OMDb", value=True)
    st.markdown("---")
    st.markdown("#### 🆔 Sample User IDs")
    sample_ids = system.all_user_ids()[:12]
    st.markdown("  ".join(f"`{u}`" for u in sample_ids))
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#333;font-size:0.7rem;line-height:1.8'>
      Built with ❤️ using<br>
      <span style='color:#E50914'>Streamlit · scikit-learn</span><br>
      Content + Collaborative + Popularity
    </div>
    """, unsafe_allow_html=True)

# ── User ID input ──────────────────────────────────────────────────────────────
user_input = st.text_input(
    "🔍 User ID",
    placeholder="Enter your User ID — e.g. 99476",
    label_visibility="visible",
)

if not user_input:
    st.markdown("""
    <div style='text-align:center;padding:5rem 0 3rem;'>
      <div style='font-size:4rem'>🎬</div>
      <div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;
                  letter-spacing:4px;color:#333;margin-top:1.2rem;'>
        ENTER YOUR USER ID ABOVE
      </div>
      <div style='color:#2A2A2A;font-size:0.85rem;margin-top:0.4rem;'>
        Your personalised movie wrap is waiting
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    user_id = int(user_input.strip())
except ValueError:
    st.error("Please enter a valid **integer** User ID.")
    st.stop()

if not system.user_exists(user_id):
    st.error(f"User **{user_id}** is not in the dataset.")
    st.stop()

username = auto_username(user_id)
st.markdown(f"""
<div style='display:flex;align-items:center;gap:1rem;
            background:#141414;border:1px solid #2A2A2A;border-radius:12px;
            padding:1rem 1.5rem;margin-bottom:1.5rem;'>
  <div style='width:48px;height:48px;border-radius:50%;background:#E50914;
              display:flex;align-items:center;justify-content:center;
              font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:white;flex-shrink:0'>
    {username[0].upper()}
  </div>
  <div>
    <div style='font-family:Bebas Neue,sans-serif;font-size:1.3rem;
                color:white;letter-spacing:2px'>@{username}</div>
    <div style='font-size:0.75rem;color:#555;font-family:Inter,sans-serif'>
      User ID {user_id} &nbsp;·&nbsp; Your CineWrap 2026
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load results ───────────────────────────────────────────────────────────────
with st.spinner("Crunching your taste profile…"):
    recent = system.get_recent_activity(user_id, top_n=top_n_recent)
    recs   = system.recommend(user_id, top_n=top_n_recs)
    genres = system.get_user_profile(user_id)

# ── Recent Activity ────────────────────────────────────────────────────────────
st.markdown("## 📽️ YOUR RECENT PHASE")
if recent.empty:
    st.warning("No watch history found for this user.")
else:
    n     = len(recent)
    cols  = min(n, 5)
    cards = [recent_card(row, show_posters) for _, row in recent.iterrows()]
    render_grid(cards, cols=cols)

# ── Recommendations ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🍿 PICKED FOR YOU")
if recs.empty:
    st.warning("Not enough data to generate recommendations.")
else:
    cards = [rec_card(row, i + 1, show_posters) for i, (_, row) in enumerate(recs.iterrows())]
    all_cards = "".join(cards)
    st.markdown(
        f'<div class="movie-grid cols-5">{all_cards}</div>',
        unsafe_allow_html=True,
    )

# ── Taste DNA ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🧬 YOUR TASTE DNA")
if genres:
    genre_df = pd.DataFrame(genres, columns=["Genre", "Score"]).set_index("Genre")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(genre_df, use_container_width=True, color="#E50914")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        for genre, score in genres:
            bar_w = int(score * 100)
            st.markdown(f"""
            <div style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;
                          font-size:0.78rem;color:#BBB;margin-bottom:3px;font-family:Inter,sans-serif'>
                <span>{genre}</span>
                <span style='color:#E50914;font-weight:700'>{score:.0%}</span>
              </div>
              <div style='background:#1C1C1C;border-radius:4px;height:5px;overflow:hidden'>
                <div style='background:linear-gradient(90deg,#B20710,#E50914);
                            width:{bar_w}%;height:5px;border-radius:4px'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Could not build a genre profile for this user.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.5rem;
            color:#222;font-size:0.72rem;font-family:Inter,sans-serif;letter-spacing:2px'>
  CINEWRAP 2026 &nbsp;·&nbsp; CONTENT + COLLABORATIVE + POPULARITY
</div>
""", unsafe_allow_html=True)
