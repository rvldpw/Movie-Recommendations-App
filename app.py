import streamlit as st
import pandas as pd
import requests
import random
from collections import Counter

from src.data_loader import load_data
from src.recommender import RecommenderSystem

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineWrap 2026",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (unchanged) ────────────────────────────────────────────────────────────
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

.movie-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.movie-grid.cols-1 { grid-template-columns: repeat(1, 1fr); }
.movie-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.movie-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.movie-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.movie-grid.cols-5 { grid-template-columns: repeat(5, 1fr); }

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
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    line-height: 1.35;
}
.poster-rating     { color: var(--gold); font-size: 0.78rem; }
.poster-date       { color: var(--muted); font-size: 0.73rem; }
.poster-user-rating{ color: #BBB; font-size: 0.75rem; }
.poster-match      { color: var(--red); font-weight: 700; font-size: 0.8rem; margin-top: auto; padding-top: 6px; }
.poster-badge {
    display: inline-block;
    background: var(--red);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    width: fit-content;
}
.country-tag {
    display: inline-block;
    background: #1a2a3a;
    color: #9ECAFF !important;
    font-size: 0.68rem;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #1e3a5a;
    width: fit-content;
    margin-top: 2px;
}
.boost-tag {
    color: #4CAF50 !important;
    font-size: 0.68rem;
    margin-top: 2px;
}

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
    content: '';
    position: absolute; top: 0; right: 0; width: 40%; height: 100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(229,9,20,0.15), transparent 70%);
    pointer-events: none;
}
.year-pill {
    display: inline-block;
    background: var(--red);
    color: white;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 3px;
    padding: 3px 14px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    color: var(--white);
    letter-spacing: 6px;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: var(--red); }
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── OMDb improved with year extraction ─────────────────────────────────────────
OMDB_API_KEY = "trilogy"
PLACEHOLDER  = "https://placehold.co/400x600/141414/333333?text=No+Poster"

# Helper to split title and year
def split_title_year(title_str: str):
    """
    Returns (clean_title, year_or_None)
    e.g. "Fight Club (1999)" -> ("Fight Club", "1999")
         "Idiocracy"         -> ("Idiocracy", None)
    """
    title_str = title_str.strip()
    if '(' in title_str and ')' in title_str:
        # Find last occurrence of '('...')' assuming it's the year
        start = title_str.rfind('(')
        end = title_str.rfind(')')
        if start != -1 and end != -1 and end > start:
            year_candidate = title_str[start+1:end].strip()
            if year_candidate.isdigit() and len(year_candidate) == 4:
                clean_title = title_str[:start].strip()
                return clean_title, year_candidate
    return title_str, None

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_info(title: str) -> dict:
    """
    Fetch movie info from OMDb using title and optional year.
    Returns dict with poster, imdb, year, genre, country.
    """
    clean_title, year = split_title_year(title)
    params = {"t": clean_title, "apikey": OMDB_API_KEY}
    if year:
        params["y"] = year

    try:
        r = requests.get("https://www.omdbapi.com/", params=params, timeout=4)
        data = r.json()
        if data.get("Response") == "True":
            poster_url = data.get("Poster", "")
            # Sometimes OMDb returns "N/A" or an empty string
            if poster_url and poster_url != "N/A":
                poster = poster_url
            else:
                poster = ""
            return {
                "poster":  poster,
                "imdb":    data.get("imdbRating", "N/A"),
                "year":    data.get("Year", ""),
                "genre":   data.get("Genre", ""),
                "country": data.get("Country", ""),
            }
    except Exception:
        pass
    return {"poster": "", "imdb": "N/A", "year": "", "genre": "", "country": ""}

def safe_poster(title: str):
    """Return (poster_url, imdb_rating) with fallback placeholder."""
    info = fetch_movie_info(title)
    url = info["poster"] if info["poster"] else PLACEHOLDER
    return url, info["imdb"]

def get_movie_country(title: str) -> str:
    """Return primary country string for the movie."""
    info = fetch_movie_info(title)
    raw = info.get("country", "") or ""
    return raw.split(",")[0].strip() if raw else ""

COUNTRY_FLAGS = {
    "USA": "🇺🇸", "United States": "🇺🇸", "UK": "🇬🇧", "United Kingdom": "🇬🇧",
    "France": "🇫🇷", "Germany": "🇩🇪", "Italy": "🇮🇹", "Spain": "🇪🇸",
    "Japan": "🇯🇵", "South Korea": "🇰🇷", "China": "🇨🇳", "India": "🇮🇳",
    "Australia": "🇦🇺", "Canada": "🇨🇦", "Mexico": "🇲🇽", "Brazil": "🇧🇷",
    "Sweden": "🇸🇪", "Denmark": "🇩🇰", "Norway": "🇳🇴", "Finland": "🇫🇮",
    "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Switzerland": "🇨🇭", "Austria": "🇦🇹",
    "Poland": "🇵🇱", "Czech Republic": "🇨🇿", "Hungary": "🇭🇺", "Romania": "🇷🇴",
    "Russia": "🇷🇺", "Turkey": "🇹🇷", "Iran": "🇮🇷", "Israel": "🇮🇱",
    "Argentina": "🇦🇷", "Colombia": "🇨🇴", "Chile": "🇨🇱", "Hong Kong": "🇭🇰",
    "Taiwan": "🇹🇼", "Thailand": "🇹🇭", "Indonesia": "🇮🇩", "Philippines": "🇵🇭",
    "Malaysia": "🇲🇾", "New Zealand": "🇳🇿", "South Africa": "🇿🇦",
    "Nigeria": "🇳🇬", "Egypt": "🇪🇬", "Greece": "🇬🇷", "Portugal": "🇵🇹",
    "Ireland": "🇮🇪",
}

def country_flag(country_str: str) -> str:
    if not country_str or country_str == "N/A":
        return "🌍"
    primary = country_str.split(",")[0].strip()
    return COUNTRY_FLAGS.get(primary, "🌍")

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
                "poster":  data.get("Poster", ""),
                "imdb":    data.get("imdbRating", "N/A"),
                "year":    data.get("Year", ""),
                "genre":   data.get("Genre", ""),
                "country": data.get("Country", ""),
            }
    except Exception:
        pass
    return {"poster": "", "imdb": "N/A", "year": "", "genre": "", "country": ""}

def safe_poster(title: str):
    info = fetch_movie_info(title)
    url  = info["poster"] if info["poster"] not in ("", "N/A") else PLACEHOLDER
    return url, info["imdb"]

def get_movie_country(title: str) -> str:
    info = fetch_movie_info(title)
    raw  = info.get("country", "") or ""
    return raw.split(",")[0].strip() if raw else ""

# ── Country profile (unchanged) ────────────────────────────────────────────────
def build_country_profile(recent_df: pd.DataFrame) -> dict:
    counter: Counter = Counter()
    for _, row in recent_df.iterrows():
        country = get_movie_country(row["title"])
        if country:
            weight = float(row.get("rating", 3)) / 5.0
            counter[country] += weight
    if not counter:
        return {}
    total = sum(counter.values())
    return {c: round(w / total, 4) for c, w in counter.most_common()}

def country_boost(country: str, profile: dict) -> float:
    if not profile or not country:
        return 1.0
    weight = profile.get(country, 0.0)
    max_w  = max(profile.values()) if profile else 1.0
    return 1.0 + 0.5 * (weight / max_w) if max_w > 0 else 1.0

def rerank_with_country(recs_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    if recs_df.empty or not profile:
        if "country" not in recs_df.columns:
            recs_df = recs_df.copy()
            recs_df["country"] = ""
            recs_df["boost"]   = 1.0
        return recs_df
    countries, boosts, new_scores = [], [], []
    for _, row in recs_df.iterrows():
        c     = get_movie_country(row["title"])
        boost = country_boost(c, profile)
        countries.append(c)
        boosts.append(boost)
        new_scores.append(row["score"] * boost)
    result = recs_df.copy()
    result["country"] = countries
    result["boost"]   = boosts
    result["score"]   = new_scores
    return result.sort_values("score", ascending=False).reset_index(drop=True)

# ── Username generator (unchanged) ─────────────────────────────────────────────
_USER_ADJ = [
    "Silent","Cosmic","Neon","Crimson","Golden","Shadow","Electric","Frozen","Blazing","Midnight",
    "Velvet","Phantom","Stellar","Iron","Solar","Lunar","Obsidian","Silver","Radiant","Quantum",
    "Cyber","Digital","Nova","Eternal","Vivid","Chrome","Emerald","Scarlet","Ivory","Onyx",
    "Atomic","Turbo","Mystic","Infernal","Arctic","Celestial","Aurora","Storm","Titan","Hyper",
    "Echo","Mirage","Voltage","Gravity","Drift","Zenith","Pulse","Turbocharged","Glitch","Alpha",
    "Omega","Prime","Astral","Molten","Hollow","Spectral","Arcane","Wild","Savage","Rogue",
    "Stealth","Dark","Bright","Crystal","Solaris","Monolith","Feral","Tempest","Nimbus","Ember",
    "Void","Binary","Static","Velocity","Shattered","Royal","Majestic","Venom","Bullet","Rapid",
    "TurboX","Nitro","Dusk","Dawn","Twilight","Monsoon","Thunder","Inferno","Icebound","Orbit",
    "Apex","Pixel","Retro","Future","Cobalt","Sapphire","Hazel","Graphite","Mercury","Plasma"
]

_USER_NOUN = [
    "Watcher","Cinephile","Director","Critic","Reel","Projector","Auteur","Curator","Maverick","Pioneer",
    "Voyager","Lens","Frame","Nomad","Explorer","Visionary","Drifter","Sentinel","Seeker","Oracle",
    "Architect","Producer","Editor","Storyteller","Screenwriter","Filmmaker","Operator","Cameraman","Animator","Collector",
    "Archivist","Dreamer","Traveler","Ranger","Pilot","Navigator","Spectator","Observer","Hunter","Wanderer",
    "Prophet","Guardian","Scholar","Strategist","Creator","Inventor","Engine","Cipher","Decoder","Signal",
    "Broadcast","Frequency","Satellite","Matrix","Protocol","System","Vector","Code","Kernel","Circuit",
    "Pixel","Render","Shader","Portal","Machine","Core","Engineer","Catalyst","Avatar","Titan",
    "Knight","Samurai","Ronin","Warrior","Ghost","Specter","Phantom","Shadow","Wolf","Falcon",
    "Dragon","Phoenix","Leviathan","Vortex","Comet","Meteor","Orbit","Nova","Galaxy","Cosmos",
    "Voyage","Dimension","Chronicle","Myth","Legend","Monarch","Empire","Frontier","Rebel","Outlaw"
]

def auto_username(user_id: int) -> str:
    rng = random.Random(int(user_id))
    return f"{rng.choice(_USER_ADJ)}{rng.choice(_USER_NOUN)}{user_id % 100:02d}"

# ── Card helpers (unchanged) ───────────────────────────────────────────────────
def _card_wrap(inner: str) -> str:
    return '<div class="poster-card">' + inner + '</div>'

def recent_card(row, show_posters: bool) -> str:
    img, imdb = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")
    country   = get_movie_country(row["title"])
    flag      = country_flag(country)
    rating    = int(round(row["rating"]))
    stars     = "&#9733;" * rating + "&#9734;" * (5 - rating)
    country_html = (
        '<div class="country-tag">' + flag + " " + country + "</div>"
        if country else ""
    )
    inner = (
        '<div class="poster-img-wrap">'
        '<img src="' + img + '" alt="" onerror="this.src=&quot;' + PLACEHOLDER + '&quot;"/>'
        '</div>'
        '<div class="poster-info">'
        '<div class="poster-title">' + row["title"] + "</div>"
        '<div class="poster-rating">' + stars + " &nbsp; IMDB " + imdb + "</div>"
        '<div class="poster-date">&#128197; ' + str(row["datetime"].date()) + "</div>"
        + country_html
        + '<div class="poster-user-rating">Your rating: ' + str(row["rating"]) + "/5</div>"
        "</div>"
    )
    return _card_wrap(inner)

def rec_card(row, rank: int, show_posters: bool) -> str:
    img, imdb = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")
    pct       = int(row["score"] * 100)
    country   = str(row.get("country") or get_movie_country(row["title"]))
    flag      = country_flag(country)
    boosted   = float(row.get("boost", 1.0)) > 1.05
    country_html = (
        '<div class="country-tag">' + flag + " " + country + "</div>"
        if country else ""
    )
    boost_html = (
        '<div class="boost-tag">&#10022; Country match</div>'
        if boosted else ""
    )
    inner = (
        '<div class="poster-img-wrap">'
        '<img src="' + img + '" alt="" onerror="this.src=&quot;' + PLACEHOLDER + '&quot;"/>'
        '</div>'
        '<div class="poster-info">'
        '<div class="poster-badge">#' + str(rank) + "</div>"
        '<div class="poster-title">' + row["title"] + "</div>"
        '<div class="poster-rating">IMDB ' + imdb + "</div>"
        + country_html
        + boost_html
        + '<div class="poster-match">&#127919; ' + str(pct) + "% match</div>"
        "</div>"
    )
    return _card_wrap(inner)

def render_grid(cards: list, cols: int = 5) -> None:
    inner = "".join(cards)
    st.markdown(
        '<div class="movie-grid cols-' + str(cols) + '">' + inner + "</div>",
        unsafe_allow_html=True,
    )

# ── Load system (unchanged) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Starting up CineWrap…")
def load_system() -> RecommenderSystem:
    df    = load_data()
    model = RecommenderSystem(df)
    model.fit()
    return model

system = load_system()

# ── Hero (unchanged) ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="year-pill">2026 EDITION</div>
  <div class="hero-title">CINE<span>WRAP</span></div>
  <div class="hero-sub">Your taste. Your universe. Mapped.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar (unchanged) ────────────────────────────────────────────────────────
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
    st.markdown("#### 🌍 Country Filter")
    country_boost_enabled = st.toggle("Boost by country preference", value=True)
    st.markdown("---")
    st.markdown("#### 🆔 Sample User IDs")
    sample_ids = sorted(random.sample(system.all_user_ids(), min(12, len(system.all_user_ids()))))
    st.markdown("  ".join(f"`{u}`" for u in sample_ids))
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#333;font-size:0.7rem;line-height:1.8'>
      Built with &#10084;&#65039
    </div>
    """, unsafe_allow_html=True)

# ========================= SEARCH FORM WITH BUTTON ==============================
st.markdown("#### 🔍 Find your CineWrap")

# Use a form so that both Enter and the button trigger the search
with st.form(key="user_search_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "User ID",
            placeholder="e.g. 99476",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

# ================ Handle empty / invalid input ==================================
if not submitted or not user_input.strip():
    # Show the animated placeholder (exactly as before)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500&display=swap');
    .cw-stage{min-height:460px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 1rem;position:relative;overflow:hidden;background:transparent}
    .cw-reel-row{display:flex;align-items:center;gap:0;margin-bottom:2.4rem;overflow:hidden;width:100%;-webkit-mask-image:linear-gradient(90deg,transparent,#000 60px,#000 calc(100% - 60px),transparent);mask-image:linear-gradient(90deg,transparent,#000 60px,#000 calc(100% - 60px),transparent)}
    .cw-reel-track{display:flex;gap:0;animation:cwReelMove 5s linear infinite}
    .cw-reel-hole{width:14px;height:22px;background:#1a1a1a;border:1px solid #333;border-radius:3px;flex-shrink:0}
    .cw-reel-cell{width:56px;height:38px;background:#141414;border:1px solid #2a2a2a;flex-shrink:0;position:relative;overflow:hidden}
    .cw-reel-cell::after{content:'';position:absolute;inset:5px;background:#0f0f0f;border-radius:1px}
    .cw-reel-cell.cw-lit{border-color:#E50914}
    .cw-reel-cell.cw-lit::after{background:#1a0000}
    @keyframes cwReelMove{0%{transform:translateX(0)}100%{transform:translateX(-71px)}}
    .cw-icon-stage{position:relative;width:100px;height:100px;margin-bottom:2rem;flex-shrink:0}
    .cw-ring{position:absolute;inset:0;border-radius:50%;border:1px solid #333;animation:cwRingSpin 12s linear infinite}
    .cw-ring.r2{animation-duration:7s;animation-direction:reverse;inset:8px;border-color:#E50914;border-width:1px;border-style:dashed;opacity:0.6}
    .cw-ring.r3{animation-duration:20s;inset:16px;border-color:#444}
    @keyframes cwRingSpin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
    .cw-icon-center{position:absolute;inset:22px;border-radius:50%;background:#141414;border:1px solid #333;display:flex;align-items:center;justify-content:center}
    .cw-play{width:0;height:0;border-top:10px solid transparent;border-bottom:10px solid transparent;border-left:18px solid #E50914;margin-left:3px;animation:cwPlayPulse 2s ease-in-out infinite}
    @keyframes cwPlayPulse{0%,100%{opacity:0.8}50%{opacity:1}}
    .cw-title{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:8px;color:#555;margin-bottom:0.35rem;animation:cwTitleBreathe 3.5s ease-in-out infinite}
    @keyframes cwTitleBreathe{0%,100%{color:#444;letter-spacing:8px}50%{color:#777;letter-spacing:9px}}
    .cw-sub{font-family:'Inter',sans-serif;font-size:0.74rem;letter-spacing:3px;color:#444;text-transform:uppercase;margin-bottom:2rem;animation:cwSubBreathe 3.5s ease-in-out infinite alternate}
    @keyframes cwSubBreathe{from{opacity:0.6}to{opacity:1}}
    .cw-poster-strip{width:100%;overflow:hidden;margin-bottom:0;-webkit-mask-image:linear-gradient(90deg,transparent,#000 80px,#000 calc(100% - 80px),transparent);mask-image:linear-gradient(90deg,transparent,#000 80px,#000 calc(100% - 80px),transparent)}
    .cw-poster-track{display:flex;gap:10px;animation:cwPostersScroll 18s linear infinite}
    .cw-poster-track.reverse{animation-direction:reverse;animation-duration:22s}
    .cw-poster{width:72px;height:104px;flex-shrink:0;border-radius:6px;background:#141414;border:1px solid #2a2a2a;position:relative;overflow:hidden;opacity:0;animation:cwPosterIn 0.4s ease forwards}
    .cw-poster.tall{width:64px;height:92px}
    .cw-poster.accent{border-color:#E50914}
    .cw-poster-shimmer{position:absolute;top:0;left:-100%;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(229,9,20,0.06),transparent);animation:cwShimmer 2.5s ease-in-out infinite}
    .cw-poster-inner{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;padding:6px}
    .cw-poster-bar{width:100%;height:3px;border-radius:2px;background:#E50914;margin-bottom:4px;transform:scaleX(0);transform-origin:left;animation:cwBarReveal 0.5s ease forwards}
    .cw-poster.accent .cw-poster-bar{background:rgba(255,255,255,0.12)}
    .cw-poster-line{width:60%;height:2px;border-radius:1px;background:#2a2a2a}
    .cw-poster-line2{width:40%;height:2px;border-radius:1px;background:#222;margin-top:3px}
    @keyframes cwShimmer{0%{left:-60%}100%{left:120%}}
    @keyframes cwBarReveal{to{transform:scaleX(1)}}
    @keyframes cwPosterIn{to{opacity:1}}
    @keyframes cwPostersScroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
    .cw-ticker-shell{width:100%;overflow:hidden;-webkit-mask-image:linear-gradient(90deg,transparent,#000 50px,#000 calc(100% - 50px),transparent);mask-image:linear-gradient(90deg,transparent,#000 50px,#000 calc(100% - 50px),transparent)}
    .cw-ticker-track{display:flex;white-space:nowrap;animation:cwTick 22s linear infinite}
    .cw-t-item{font-family:'Bebas Neue',sans-serif;font-size:0.72rem;letter-spacing:3px;color:#333;padding:0 16px}
    .cw-t-dot{color:#E50914;opacity:0.5;font-size:0.5rem;padding:0 2px}
    @keyframes cwTick{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
    .cw-particle{position:absolute;border-radius:50%;pointer-events:none;opacity:0;background:#E50914}
    @keyframes cwPFly{0%{opacity:0;transform:translate(0,0) scale(0)}10%{opacity:1}80%{opacity:0.5}100%{opacity:0;transform:translate(var(--dx),var(--dy)) scale(0.4)}}
    </style>

    <div class="cw-stage" id="cwStage">
      <div class="cw-reel-row"><div class="cw-reel-track" id="cwReelTrack"></div></div>

      <div class="cw-icon-stage" id="cwIconStage">
        <div class="cw-ring"></div>
        <div class="cw-ring r2"></div>
        <div class="cw-ring r3"></div>
        <div class="cw-icon-center"><div class="cw-play"></div></div>
      </div>

      <div class="cw-title">ENTER YOUR USER ID</div>
      <div class="cw-sub">Your personalised movie wrap is waiting</div>

      <div class="cw-poster-strip"><div class="cw-poster-track" id="cwPosterRow1"></div></div>
      <div class="cw-poster-strip" style="margin-top:-1.2rem"><div class="cw-poster-track reverse" id="cwPosterRow2"></div></div>

      <div class="cw-ticker-shell" style="margin-top:1.2rem"><div class="cw-ticker-track" id="cwTicker"></div></div>
    </div>

    <script>
    (function(){
      var reel=document.getElementById('cwReelTrack');
      if(reel){
        for(var i=0;i<24;i++){
          var h=document.createElement('div');h.className='cw-reel-hole';reel.appendChild(h);
          var c=document.createElement('div');c.className='cw-reel-cell'+(i%3===1?' cw-lit':'');reel.appendChild(c);
        }
        var hx=document.createElement('div');hx.className='cw-reel-hole';reel.appendChild(hx);
      }

      var heights=['','tall',''];
      var accents=[false,false,true,false,false,false,true,false,false,false,true,false,false,false,false,true];

      function makePosters(id,count){
        var row=document.getElementById(id);
        if(!row)return;
        for(var i=0;i<count*2;i++){
          var cls='cw-poster'+(heights[i%3]?' '+heights[i%3]:'')+(accents[i%accents.length]?' accent':'');
          var delay=((i%count)*0.07).toFixed(2);
          var barDelay=(parseFloat(delay)+0.15).toFixed(2);
          var div=document.createElement('div');
          div.className=cls;
          div.style.animationDelay=delay+'s';
          div.innerHTML='<div class="cw-poster-shimmer"></div><div class="cw-poster-inner"><div class="cw-poster-bar" style="animation-delay:'+barDelay+'s"></div><div class="cw-poster-line"></div><div class="cw-poster-line2"></div></div>';
          row.appendChild(div);
        }
      }
      makePosters('cwPosterRow1',14);
      makePosters('cwPosterRow2',14);

      var ticker=document.getElementById('cwTicker');
      if(ticker){
        var films=['PARASITE','INCEPTION','SPIRITED AWAY','THE GODFATHER','OLDBOY','MULHOLLAND DRIVE','AMÉLIE','BLADE RUNNER 2049','2001: A SPACE ODYSSEY','GOODFELLAS','PRINCESS MONONOKE','THE DARK KNIGHT','CITY OF GOD','HEAT','FARGO','AKIRA','NOSFERATU','METROPOLIS','WILD STRAWBERRIES','BREATHLESS'];
        var doubled=films.concat(films);
        doubled.forEach(function(f,i){
          var s=document.createElement('span');s.className='cw-t-item';s.textContent=f;ticker.appendChild(s);
          if(i<doubled.length-1){var d=document.createElement('span');d.className='cw-t-dot';d.textContent='&#9670;';ticker.appendChild(d);}
        });
      }

      var stage=document.getElementById('cwStage');
      var is=document.getElementById('cwIconStage');
      if(stage&&is){
        function getCenter(){
          var sr=stage.getBoundingClientRect();
          var ir=is.getBoundingClientRect();
          return{x:ir.left-sr.left+50,y:ir.top-sr.top+50};
        }
        setInterval(function(){
          var c=getCenter();
          var p=document.createElement('div');p.className='cw-particle';
          var angle=Math.random()*Math.PI*2;
          var dist=70+Math.random()*100;
          var dx=(Math.cos(angle)*dist).toFixed(1);
          var dy=(Math.sin(angle)*dist).toFixed(1);
          var sz=(2+Math.random()*3).toFixed(1);
          var dur=(1.6+Math.random()*1.6).toFixed(2);
          p.style.cssText='width:'+sz+'px;height:'+sz+'px;left:'+(c.x+Math.cos(angle)*10).toFixed(1)+'px;top:'+(c.y+Math.sin(angle)*10).toFixed(1)+'px;--dx:'+dx+'px;--dy:'+dy+'px;animation:cwPFly '+dur+'s ease-out forwards';
          stage.appendChild(p);
          setTimeout(function(){if(p.parentNode)p.parentNode.removeChild(p);},parseFloat(dur)*1000+100);
        },180);

        var dots=[];
        for(var i=0;i<10;i++){
          var dot=document.createElement('div');
          dot.style.cssText='width:4px;height:4px;border-radius:50%;background:#E50914;position:absolute;top:50%;left:50%;margin:-2px;pointer-events:none;';
          is.appendChild(dot);
          dots.push({el:dot,offset:(i/10)*Math.PI*2});
        }
        var frame=0;
        function animDots(){
          frame+=0.018;
          for(var i=0;i<dots.length;i++){
            var a=dots[i].offset+frame;
            var x=(Math.cos(a)*40).toFixed(2),y=(Math.sin(a)*40).toFixed(2);
            dots[i].el.style.transform='translate(calc(-50% + '+x+'px),calc(-50% + '+y+'px))';
            dots[i].el.style.opacity=(0.3+0.7*((Math.sin(a*2+i)*0.5)+0.5)).toFixed(2);
          }
          requestAnimationFrame(animDots);
        }
        animDots();
      }
    })();
    </script>
    """, unsafe_allow_html=True)
    st.stop()

# ================ Process valid user ID =========================================
try:
    user_id = int(user_input.strip())
except ValueError:
    st.error("Please enter a valid **integer** User ID.")
    st.stop()

if not system.user_exists(user_id):
    st.error(f"User **{user_id}** not found. Please create a Netflix account at https://www.netflix.com/.")
    st.stop()

# Username & profile header
username = auto_username(user_id)
st.markdown(
    "<div style='display:flex;align-items:center;gap:1rem;"
    "background:#141414;border:1px solid #2A2A2A;border-radius:12px;"
    "padding:1rem 1.5rem;margin-bottom:1.5rem;'>"
    "<div style='width:48px;height:48px;border-radius:50%;background:#E50914;"
    "display:flex;align-items:center;justify-content:center;"
    "font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:white;flex-shrink:0'>"
    + username[0].upper()
    + "</div>"
    "<div>"
    "<div style='font-family:Bebas Neue,sans-serif;font-size:1.3rem;"
    "color:white;letter-spacing:2px'>@" + username + "</div>"
    "<div style='font-size:0.75rem;color:#555;font-family:Inter,sans-serif'>"
    "User ID " + str(user_id) + " &nbsp;&middot;&nbsp; Your CineWrap 2026"
    "</div></div></div>",
    unsafe_allow_html=True,
)

# ── Load results ───────────────────────────────────────────────────────────────
with st.spinner("Crunching your taste profile…"):
    recent = system.get_recent_activity(user_id, top_n=top_n_recent)
    recs   = system.recommend(user_id, top_n=top_n_recs)
    genres = system.get_user_profile(user_id)

with st.spinner("🌍 Learning your country preferences…"):
    full_history    = system.get_recent_activity(user_id, top_n=200)
    country_profile = build_country_profile(full_history) if country_boost_enabled else {}

# ── Country DNA in sidebar (unchanged) ─────────────────────────────────────────
if country_profile:
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🌍 Your Country DNA")
        for c, w in list(country_profile.items())[:6]:
            flag = country_flag(c)
            bar  = int(w * 100)
            st.markdown(
                "<div style='margin-bottom:6px'>"
                "<div style='display:flex;justify-content:space-between;"
                "font-size:0.75rem;color:#BBB;margin-bottom:2px'>"
                "<span>" + flag + " " + c + "</span>"
                "<span style='color:#E50914'>" + str(bar) + "%</span>"
                "</div>"
                "<div style='background:#1C1C1C;border-radius:3px;height:4px'>"
                "<div style='background:#E50914;width:" + str(bar) + "%;height:4px;border-radius:3px'></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

# ── Recommendations (unchanged) ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🍿 PICKED FOR YOU")

if recs.empty:
    st.warning("Not enough data to generate recommendations.")
else:
    with st.spinner("🌍 Applying country preferences…"):
        recs_ranked = rerank_with_country(recs, country_profile)

    if country_profile:
        top_country = max(country_profile, key=country_profile.get)
        flag        = country_flag(top_country)
        st.markdown(
            "<p style='color:#555;font-size:0.8rem;margin-top:-0.5rem;'>"
            + flag + " Personalised for your love of "
            "<strong style='color:#9ECAFF'>" + top_country + "</strong>"
            " cinema &amp; more</p>",
            unsafe_allow_html=True,
        )

    cards     = [rec_card(row, i + 1, show_posters) for i, (_, row) in enumerate(recs_ranked.iterrows())]
    all_cards = "".join(cards)
    st.markdown(
        '<div class="movie-grid cols-5">' + all_cards + "</div>",
        unsafe_allow_html=True,
    )

# ── DNA Nickname Generator ─────────────────────────────────────────────────────
DNA_ADJ = {
    "Drama":     ["Tragic","Emotive","Brooding","Melancholic","Visceral"],
    "Thriller":  ["Paranoid","Razor","Shadow","Pulse","Fractured"],
    "Action":    ["Kinetic","Furious","Titanium","Electric","Turbo"],
    "Comedy":    ["Absurdist","Deadpan","Manic","Vivid","Neon"],
    "Horror":    ["Haunted","Primordial","Void","Spectral","Infernal"],
    "Romance":   ["Velvet","Tender","Aching","Golden","Ephemeral"],
    "Sci-Fi":    ["Quantum","Orbital","Binary","Stellar","Synthetic"],
    "Crime":     ["Noir","Rogue","Cryptic","Obsidian","Hollow"],
    "Animation": ["Dreaming","Whimsical","Vivid","Boundless","Surreal"],
    "Documentary":["Lucid","Raw","Unfiltered","Searching","Honest"],
    "Fantasy":   ["Mythic","Arcane","Wandering","Enchanted","Ancient"],
    "Adventure": ["Nomadic","Drifting","Boundless","Fearless","Wild"],
}
DNA_NOUN = ["Auteur","Cinephile","Curator","Visionary","Archivist",
            "Oracle","Phantom","Pioneer","Seeker","Nomad"]

def get_dna_nickname(genres: list, user_id: int) -> tuple[str, str]:
    top_genre = genres[0][0] if genres else "Drama"
    adjs = DNA_ADJ.get(top_genre, ["Cosmic","Liminal","Drifting"])
    rng  = random.Random(user_id * 13)
    return rng.choice(adjs), rng.choice(DNA_NOUN)

def rarity_score(genres: list, user_id: int) -> int:
    if not genres: return 0
    diversity  = len(genres) * 12
    top_score  = int((1 - genres[0][1]) * 200) if genres else 0
    return min(499, diversity + top_score + (user_id % 47))

GENRE_COLORS = [
    "#E50914","#F5C518","#4CAF50","#2196F3","#FF5722",
    "#E91E63","#9C27B0","#00BCD4","#FF9800","#607D8B",
]

def genre_color(idx: int) -> str:
    return GENRE_COLORS[idx % len(GENRE_COLORS)]


# ── Recent Activity + DNA side-by-side ────────────────────────────────────────
dna_adj, dna_noun = get_dna_nickname(genres, user_id)
dna_score         = rarity_score(genres, user_id)
top_pct           = max(1, round((1 - dna_score / 500) * 100))

# Build genre data JSON for JS
genre_js = "[" + ",".join(
    f'{{"name":"{g}","score":{round(s,3)},"color":"{genre_color(i)}"}}'
    for i, (g, s) in enumerate(genres)
) + "]"

# Build recent movies JSON for JS

# Build recent movies JSON for JS
recent_js_items = []

for _, row in recent.iterrows():
    poster_url, _ = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")

    recent_js_items.append({
        "title": row["title"],
        "rating": int(round(row["rating"])),
        "date": str(row["datetime"].date()),
        "poster": poster_url
    })

# Convert safely to JavaScript/JSON
recent_js = json.dumps(recent_js_items)

st.markdown("## 🎬 YOUR CINEWRAP UNIVERSE")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
.cw-uni-row{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}}
.dna-card{{background:#0f0000;border:1px solid #2a0808;border-radius:14px;padding:14px 12px;position:relative;overflow:hidden;display:flex;flex-direction:column;min-height:290px}}
.dna-logo{{font-family:'Bebas Neue',sans-serif;font-size:10px;letter-spacing:3px;color:#E50914;margin-bottom:8px}}
.dna-arch-label{{font-size:9px;letter-spacing:2px;color:#444;text-transform:uppercase;margin-bottom:3px}}
.dna-nickname{{font-family:'Bebas Neue',sans-serif;font-size:21px;letter-spacing:2px;color:#fff;line-height:1.05;margin-bottom:10px}}
.dna-nickname span{{color:#E50914}}
.dna-helix{{flex:1;display:flex;flex-direction:column;gap:6px;margin:4px 0}}
.dna-rung{{display:flex;align-items:center;gap:4px}}
.dna-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.dna-bar{{height:3px;border-radius:2px;flex:1}}
.dna-glabel{{font-size:8px;font-family:'Space Mono',monospace;color:#555;white-space:nowrap;flex-shrink:0;font-weight:700}}
.dna-bottom{{margin-top:10px;padding-top:10px;border-top:1px solid #1a0808}}
.dna-stats{{display:flex;align-items:flex-end;justify-content:space-between}}
.dna-score{{font-family:'Bebas Neue',sans-serif;font-size:30px;color:#F5C518;line-height:1}}
.dna-score-sub{{font-size:8px;color:#333;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-top:2px}}
.dna-pct-top{{font-size:11px;color:#555;font-family:'Bebas Neue',sans-serif;letter-spacing:1.5px;text-align:right}}
.dna-pct-bot{{font-size:8px;color:#333;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;text-align:right}}
.ss-wrap{{position:relative;border-radius:14px;overflow:hidden;background:#0d0d0d;border:1px solid #2A2A2A;min-height:290px}}
.ss-slide{{position:absolute;inset:0;display:flex;flex-direction:column;opacity:0;transition:opacity .6s ease;pointer-events:none}}
.ss-slide.active{{opacity:1;pointer-events:auto}}
.ss-img{{width:100%;height:155px;background:#130000;display:flex;align-items:center;justify-content:center;flex-shrink:0;overflow:hidden}}
.ss-img img{{width:100%;height:100%;object-fit:cover}}
.ss-info{{padding:9px 10px;flex:1;display:flex;flex-direction:column;gap:3px}}
.ss-title{{font-size:11px;font-weight:600;color:#fff;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.ss-stars{{font-size:10px;color:#F5C518}}
.ss-date{{font-size:9px;color:#666;margin-top:auto}}
.ss-badge{{position:absolute;top:8px;left:8px;background:#E50914;color:white;font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;font-family:'Space Mono',monospace}}
.ss-counter{{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.6);border-radius:10px;padding:2px 7px;font-size:9px;color:#888;font-family:'Space Mono',monospace}}
.ss-dots{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);display:flex;gap:4px}}
.ss-dot{{width:5px;height:5px;border-radius:50%;background:#333;cursor:pointer;transition:background .3s,width .3s}}
.ss-dot.active{{background:#E50914;width:14px;border-radius:3px}}
.sc-canvas{{background:#0f0000;border-radius:16px;border:1px solid #2a0808;padding:20px 18px;position:relative;overflow:hidden;margin-bottom:10px}}
.sc-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}}
.sc-brand{{font-family:'Bebas Neue',sans-serif;font-size:11px;letter-spacing:4px;color:#E50914}}
.sc-year{{font-family:'Bebas Neue',sans-serif;font-size:11px;letter-spacing:3px;color:#333}}
.sc-nick{{font-family:'Bebas Neue',sans-serif;font-size:30px;letter-spacing:3px;color:#fff;line-height:1;margin-bottom:4px}}
.sc-nick span{{color:#E50914}}
.sc-type{{font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:16px}}
.sc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}}
.sc-stat{{background:rgba(255,255,255,.03);border:1px solid #1e1e1e;border-radius:8px;padding:8px 10px}}
.sc-stat-label{{font-size:8px;color:#444;letter-spacing:2px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:3px}}
.sc-stat-value{{font-family:'Bebas Neue',sans-serif;font-size:18px;color:#fff;line-height:1}}
.sc-stat-value.gold{{color:#F5C518}}
.sc-stat-value.red{{color:#E50914}}
.sc-dna-row{{display:flex;gap:3px;margin-bottom:12px}}
.sc-dna-seg{{height:4px;border-radius:2px}}
.sc-pills{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px}}
.sc-pill{{font-size:9px;font-family:'Space Mono',monospace;padding:3px 8px;border-radius:20px;font-weight:700;border:1px solid}}
.sc-footer{{display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid #1a1a1a}}
.sc-uname{{font-family:'Bebas Neue',sans-serif;font-size:13px;letter-spacing:2px;color:#555}}
.sc-tag{{font-size:9px;color:#333;font-family:'Space Mono',monospace}}
.sc-share-btn{{width:100%;background:#E50914;color:white;border:none;border-radius:8px;padding:10px;font-family:'Bebas Neue',sans-serif;font-size:14px;letter-spacing:2px;cursor:pointer}}
@media(max-width:480px){{.cw-uni-row{{grid-template-columns:1fr}}}}
</style>

<div class="cw-uni-row">
  <div>
    <div class="dna-card">
      <div class="dna-logo">◈ CINEWRAP DNA</div>
      <div class="dna-arch-label">Your movie archetype</div>
      <div class="dna-nickname">THE <span>{dna_adj.upper()}</span><br>{dna_noun.upper()}</div>
      <div class="dna-helix" id="cwDnaHelix"></div>
      <div class="dna-bottom">
        <div class="dna-stats">
          <div>
            <div class="dna-score">{dna_score}</div>
            <div class="dna-score-sub">RARITY SCORE</div>
          </div>
          <div>
            <div class="dna-pct-top">TOP {top_pct}%</div>
            <div class="dna-pct-bot">OF CINEPHILES</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div>
    <div class="ss-wrap" id="cwSsWrap">
      <div class="ss-counter" id="cwSsCounter">1 / {len(recent)}</div>
      <div id="cwSlides"></div>
      <div class="ss-dots" id="cwSsDots"></div>
    </div>
  </div>
</div>

<p style="font-family:'Bebas Neue',sans-serif;font-size:13px;letter-spacing:4px;color:#666;margin-bottom:12px;">◈ SHARE YOUR WRAP</p>
<div class="sc-canvas">
  <div class="sc-header">
    <div class="sc-brand">CINEWRAP 2026</div>
    <div class="sc-year">FILM YEAR</div>
  </div>
  <div class="sc-nick">THE <span>{dna_adj.upper()}</span> {dna_noun.upper()}</div>
  <div class="sc-type">{" · ".join(g for g, _ in genres[:3])}</div>
  <div class="sc-grid">
    <div class="sc-stat"><div class="sc-stat-label">TOP GENRE</div><div class="sc-stat-value">{genres[0][0].upper() if genres else '—'}</div></div>
    <div class="sc-stat"><div class="sc-stat-label">RARITY</div><div class="sc-stat-value gold">{dna_score}</div></div>
    <div class="sc-stat"><div class="sc-stat-label">FILMS RATED</div><div class="sc-stat-value red" id="cwFilmsRated">—</div></div>
    <div class="sc-stat"><div class="sc-stat-label">AVG RATING</div><div class="sc-stat-value" id="cwAvgRating">—</div></div>
  </div>
  <div class="sc-dna-row" id="cwScDna"></div>
  <div class="sc-pills" id="cwScPills"></div>
  <div class="sc-footer">
    <div class="sc-uname">@{username}</div>
    <div class="sc-tag">cinewrap.app #MYWRAP2026</div>
  </div>
</div>
<button class="sc-share-btn" onclick="cwShare()">↑ SHARE YOUR WRAP</button>

<script>
(function(){{
  var genres={genre_js};
  var movies={recent_js};
  var username="{username}";

  var PILL_COLORS=[
    {{bg:'rgba(229,9,20,.12)',b:'rgba(229,9,20,.4)',t:'#ff6666'}},
    {{bg:'rgba(245,197,24,.1)',b:'rgba(245,197,24,.35)',t:'#d4a017'}},
    {{bg:'rgba(76,175,80,.1)',b:'rgba(76,175,80,.35)',t:'#4CAF50'}},
    {{bg:'rgba(33,150,243,.1)',b:'rgba(33,150,243,.35)',t:'#64b5f6'}},
    {{bg:'rgba(255,87,34,.1)',b:'rgba(255,87,34,.35)',t:'#ff8a65'}},
    {{bg:'rgba(233,30,99,.1)',b:'rgba(233,30,99,.35)',t:'#f48fb1'}},
  ];

  function renderHelix(){{
    var h=document.getElementById('cwDnaHelix');
    if(!h)return;
    genres.forEach(function(g,i){{
      var r=document.createElement('div');
      r.className='dna-rung';
      r.style.paddingLeft=(i%2===0?0:10)+'px';
      var w=Math.max(15,Math.floor(g.score*75));
      r.innerHTML='<div class="dna-dot" style="background:'+g.color+';opacity:.8"></div>'
        +'<div class="dna-bar" style="background:'+g.color+';max-width:'+w+'%"></div>'
        +'<div class="dna-glabel">'+g.name.toUpperCase().slice(0,7)+'</div>'
        +'<div class="dna-dot" style="background:'+g.color+';opacity:.35"></div>';
      h.appendChild(r);
    }});
  }}

  function renderSlides(){{
    var c=document.getElementById('cwSlides');
    var d=document.getElementById('cwSsDots');
    if(!c||!d)return;
    movies.forEach(function(m,i){{
      var s=document.createElement('div');
      s.className='ss-slide'+(i===0?' active':'');
      var stars='★'.repeat(m.rating)+'☆'.repeat(5-m.rating);
      var t=m.title.length>24?m.title.slice(0,23)+'…':m.title;
      var imgHtml=m.poster&&m.poster!=='https://placehold.co/400x600/141414/333333?text=No+Poster'
        ?'<img src="'+m.poster+'" alt="" onerror="this.parentNode.innerHTML=\'🎬\'">'
        :'<span style="font-size:28px;opacity:.15">🎬</span>';
      s.innerHTML='<div class="ss-badge">#'+(i+1)+' RECENT</div>'
        +'<div class="ss-img">'+imgHtml+'</div>'
        +'<div class="ss-info">'
        +'<div class="ss-title">'+t+'</div>'
        +'<div class="ss-stars">'+stars+'</div>'
        +'<div class="ss-date">'+m.date+'</div>'
        +'</div>';
      c.appendChild(s);
      var dot=document.createElement('div');
      dot.className='ss-dot'+(i===0?' active':'');
      (function(idx){{dot.onclick=function(){{goTo(idx);}};}})(i);
      d.appendChild(dot);
    }});
    window._cwCur=0;
    clearInterval(window._cwTimer);
    window._cwTimer=setInterval(function(){{
      goTo((window._cwCur+1)%movies.length);
    }},3000);
  }}

  function goTo(idx){{
    document.querySelectorAll('.ss-slide').forEach(function(s,i){{s.classList.toggle('active',i===idx);}});
    document.querySelectorAll('.ss-dot').forEach(function(d,i){{d.classList.toggle('active',i===idx);}});
    var el=document.getElementById('cwSsCounter');
    if(el)el.textContent=(idx+1)+' / '+movies.length;
    window._cwCur=idx;
  }}

  function renderShareCard(){{
    var dnaRow=document.getElementById('cwScDna');
    if(dnaRow){{
      genres.forEach(function(g){{
        var seg=document.createElement('div');
        seg.className='sc-dna-seg';
        seg.style.background=g.color;
        seg.style.opacity=g.score;
        seg.style.flex=g.score;
        dnaRow.appendChild(seg);
      }});
    }}
    var pills=document.getElementById('cwScPills');
    if(pills){{
      genres.forEach(function(g,i){{
        var c=PILL_COLORS[i%PILL_COLORS.length];
        var p=document.createElement('span');
        p.className='sc-pill';
        p.textContent=g.name.toUpperCase();
        p.style.background=c.bg;
        p.style.borderColor=c.b;
        p.style.color=c.t;
        pills.appendChild(p);
      }});
    }}
    var ratings=movies.map(function(m){{return m.rating;}});
    var avg=ratings.length?ratings.reduce(function(a,b){{return a+b;}},0)/ratings.length:0;
    var fr=document.getElementById('cwFilmsRated');
    var ar=document.getElementById('cwAvgRating');
    if(fr)fr.textContent=movies.length;
    if(ar)ar.textContent='★ '+avg.toFixed(1);
  }}

  window.cwShare=function(){{
    var btn=document.querySelector('.sc-share-btn');
    if(!btn)return;
    btn.textContent='✓ COPIED — PASTE TO STORIES!';
    btn.style.background='#1a6b1a';
    setTimeout(function(){{btn.textContent='↑ SHARE YOUR WRAP';btn.style.background='';}},2200);
    if(navigator.clipboard)navigator.clipboard.writeText('cinewrap.app/u/'+username+' #MYWRAP2026').catch(function(){{}});
  }};

  renderHelix();
  renderSlides();
  renderShareCard();
})();
</script>
""", unsafe_allow_html=True)

# ── Footer (unchanged) ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;padding:1.5rem 0 0.5rem;"
    "color:#222;font-size:0.72rem;font-family:Inter,sans-serif;letter-spacing:2px'>"
    "CINEWRAP 2026 &nbsp;&middot;&nbsp; CONTENT + COLLABORATIVE + POPULARITY + COUNTRY"
    "</div>",
    unsafe_allow_html=True,
)
