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

# ── OMDb ───────────────────────────────────────────────────────────────────────
OMDB_API_KEY = "trilogy"
PLACEHOLDER  = "https://placehold.co/400x600/141414/333333?text=No+Poster"

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

# ── Country profile ────────────────────────────────────────────────────────────
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

# ── Username generator ─────────────────────────────────────────────────────────
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

# ── Card helpers ───────────────────────────────────────────────────────────────
def _card_wrap(inner: str) -> str:
    return (
        '<div class="poster-card">'
        + inner
        + '</div>'
    )

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

# ── User ID input ──────────────────────────────────────────────────────────────
user_input = st.text_input(
    "🔍 User ID",
    placeholder="Enter your User ID — e.g. 99476",
    label_visibility="visible",
)

if not user_input:
    st.markdown("""
    <div style='text-align:center;padding:5rem 0 3rem;'>
      <div style='font-size:4rem'>&#127916;</div>
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

# ── Country DNA in sidebar ─────────────────────────────────────────────────────
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
            st.markdown(
                "<div style='margin-bottom:8px'>"
                "<div style='display:flex;justify-content:space-between;"
                "font-size:0.78rem;color:#BBB;margin-bottom:3px;font-family:Inter,sans-serif'>"
                "<span>" + genre + "</span>"
                "<span style='color:#E50914;font-weight:700'>" + str(bar_w) + "%</span>"
                "</div>"
                "<div style='background:#1C1C1C;border-radius:4px;height:5px;overflow:hidden'>"
                "<div style='background:linear-gradient(90deg,#B20710,#E50914);"
                "width:" + str(bar_w) + "%;height:5px;border-radius:4px'></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
else:
    st.warning("Could not build a genre profile for this user.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;padding:1.5rem 0 0.5rem;"
    "color:#222;font-size:0.72rem;font-family:Inter,sans-serif;letter-spacing:2px'>"
    "CINEWRAP 2026 &nbsp;&middot;&nbsp; CONTENT + COLLABORATIVE + POPULARITY + COUNTRY"
    "</div>",
    unsafe_allow_html=True,
)
