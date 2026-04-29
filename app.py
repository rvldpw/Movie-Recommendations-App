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

# ── CSS (unchanged, same as before) ────────────────────────────────────────────
st.markdown("""
<style>
/* ... (keep all the same CSS as in the previous version) ... */
</style>
""", unsafe_allow_html=True)

# ── OMDb helpers ───────────────────────────────────────────────────────────────
OMDB_API_KEY = "trilogy"
PLACEHOLDER  = "https://placehold.co/400x600/141414/333333?text=No+Poster"

def split_title_year(title_str: str):
    title_str = title_str.strip()
    if '(' in title_str and ')' in title_str:
        start = title_str.rfind('(')
        end   = title_str.rfind(')')
        if start != -1 and end != -1 and end > start:
            year_candidate = title_str[start+1:end].strip()
            if year_candidate.isdigit() and len(year_candidate) == 4:
                return title_str[:start].strip(), year_candidate
    return title_str, None

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_info(title: str) -> dict:
    clean_title, year = split_title_year(title)
    params = {"t": clean_title, "apikey": OMDB_API_KEY}
    if year:
        params["y"] = year
    try:
        r    = requests.get("https://www.omdbapi.com/", params=params, timeout=4)
        data = r.json()
        if data.get("Response") == "True":
            poster_url = data.get("Poster", "")
            return {
                "poster":  poster_url if poster_url and poster_url != "N/A" else "",
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
    url  = info["poster"] if info["poster"] else PLACEHOLDER
    return url, info["imdb"]

def get_movie_country(title: str) -> str:
    info = fetch_movie_info(title)
    raw  = info.get("country", "") or ""
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

# ── DNA Nickname generator ─────────────────────────────────────────────────────
_DNA_ADJ = {
    "Drama":     ["Tragic", "Emotive", "Brooding", "Melancholic", "Visceral"],
    "Thriller":  ["Paranoid", "Razor", "Shadow", "Pulse", "Fractured"],
    "Romance":   ["Velvet", "Tender", "Aching", "Golden", "Ephemeral"],
    "Comedy":    ["Absurdist", "Deadpan", "Manic", "Vivid", "Neon"],
    "Horror":    ["Haunted", "Primordial", "Void", "Spectral", "Infernal"],
    "Action":    ["Kinetic", "Furious", "Titanium", "Electric", "Turbo"],
    "Sci-Fi":    ["Quantum", "Orbital", "Binary", "Stellar", "Synthetic"],
    "Crime":     ["Noir", "Rogue", "Cryptic", "Obsidian", "Hollow"],
    "Animation": ["Dreaming", "Vivid", "Boundless", "Whimsical", "Radiant"],
    "Documentary":["Searching","Unblinking","Restless","Piercing","Relentless"],
    "Adventure": ["Wandering", "Nomadic", "Boundless", "Feral", "Drifting"],
    "Fantasy":   ["Arcane", "Mythic", "Liminal", "Celestial", "Enchanted"],
    "Mystery":   ["Cryptic", "Veiled", "Hollow", "Shadowed", "Elusive"],
    "Biography": ["Measured", "Searching", "Unblinking", "Lucid", "Truthful"],
}
_DNA_NOUN = [
    "Auteur", "Cinephile", "Curator", "Visionary", "Archivist",
    "Oracle", "Phantom", "Pioneer", "Seeker", "Nomad",
    "Chronicler", "Wanderer", "Sentinel", "Dreamer", "Observer"
]

_GENRE_COLORS = [
    "#E50914", "#F5C518", "#4CAF50", "#2196F3",
    "#FF5722", "#E91E63", "#9C27B0", "#00BCD4",
]

def get_dna_nickname(genres: list, user_id: int) -> tuple:
    top_genre = genres[0][0] if genres else "Drama"
    adjs = _DNA_ADJ.get(top_genre, ["Cosmic", "Liminal", "Wandering"])
    rng  = random.Random(int(user_id) * 13)
    return rng.choice(adjs), rng.choice(_DNA_NOUN)

def rarity_score(genres: list, user_id: int) -> int:
    if not genres:
        return 100
    scores    = [s for _, s in genres]
    diversity = len(genres) * 11
    top_score = int((1 - scores[0]) * 180) if scores else 0
    base      = diversity + top_score + (int(user_id) % 61)
    return min(499, max(50, base))

# ── Card helpers ───────────────────────────────────────────────────────────────
def _card_wrap(inner: str) -> str:
    return '<div class="poster-card">' + inner + '</div>'

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
    top_n_recs = st.selectbox(
        "How many picks?",
        options=[10, 20, 30, 40, 50],
        index=0,
        help="Number of movie recommendations (multiples of 10)"
    )
    st.markdown("#### 📅 Watch History")
    top_n_recent = st.selectbox(
        "Recent movies shown",
        options=[5, 10, 15, 20, 25],
        index=0,
        help="Number of recent movies to display (multiples of 5)"
    )
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
      Built with &#10084;&#65039;
    </div>
    """, unsafe_allow_html=True)

# ── Search form ────────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Find your CineWrap")

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

# ── Empty state (same as before, omitted for brevity) ─────────────────────────
if not submitted or not user_input.strip():
    st.markdown("""... (same empty state HTML) ...""", unsafe_allow_html=True)
    st.stop()

# ── Validate user ID ───────────────────────────────────────────────────────────
try:
    user_id = int(user_input.strip())
except ValueError:
    st.error("Please enter a valid **integer** User ID.")
    st.stop()

if not system.user_exists(user_id):
    st.error(f"User **{user_id}** not found. Please create a Netflix account at https://www.netflix.com/.")
    st.stop()

# ── Profile header ─────────────────────────────────────────────────────────────
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
    "<div style='font-size:0.75rem;color:#555;font-family:DM Sans,sans-serif'>"
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

# ══════════════════════════════════════════════════════════════════════════════
# 🧬 DNA CARD + 📽️ RECENT SLIDESHOW  (50/50 side by side)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🧬 YOUR CINEWRAP UNIVERSE")

# Build DNA data
dna_adj, dna_noun = get_dna_nickname(genres, user_id)
score             = rarity_score(genres, user_id)
pct_top           = max(1, min(99, round((1 - score / 500) * 100)))
genre_colors_list = _GENRE_COLORS[:len(genres)]

# ── DNA helix rungs ────────────────────────────────────────────────────────────
helix_rungs = ""
for i, (genre_name, genre_score) in enumerate(genres):
    color   = genre_colors_list[i] if i < len(genre_colors_list) else "#888"
    bar_w   = max(20, int(genre_score * 75))
    label   = genre_name.upper()[:7]
    helix_rungs += (
        f'<div class="dna-rung">'
        f'<div class="dna-dot-l" style="background:{color};opacity:.9"></div>'
        f'<div class="dna-bar" style="background:{color};opacity:0.85;flex:{bar_w}"></div>'
        f'<div class="dna-bar" style="background:transparent;flex:{100 - bar_w}"></div>'
        f'<div class="dna-genre-label">{label}</div>'
        f'<div class="dna-dot-r" style="background:{color};opacity:.3"></div>'
        f'</div>'
    )
# ── Build slides data ──────────────────────────────────────────────────────────
slides_html = ""
total_slides = len(recent) if not recent.empty else 0
use_slideshow = total_slides > 1   # slideshow for 2+ movies

for i, (_, row) in enumerate(recent.iterrows()):
    poster_url, imdb_r = safe_poster(row["title"]) if show_posters else (PLACEHOLDER, "N/A")
    rating_val  = int(round(row["rating"]))
    stars       = "★" * rating_val + "☆" * (5 - rating_val)
    country     = get_movie_country(row["title"])
    flag        = country_flag(country)
    country_html = (
        f'<div class="country-tag" style="margin-top:6px">{flag} {country}</div>'
        if country else ""
    )
    active_cls = " active" if i == 0 else ""
    slides_html += (
        f'<div class="slide{active_cls}" data-idx="{i}">'
        f'  <div class="slide-poster">'
        f'    <img src="{poster_url}" alt="" onerror="this.src=\'{PLACEHOLDER}\'"/>'
        f'  </div>'
        f'  <div class="slide-info">'
        f'    <div class="slide-title">{row["title"]}</div>'
        f'    <div class="slide-rating">{stars}</div>'
        f'    <div class="slide-imdb-badge">IMDb {imdb_r}</div>'
        f'    <div class="slide-date">📅 {str(row["datetime"].date())}</div>'
        f'    {country_html}'
        f'    <div style="flex:1"></div>'
        f'    <div class="slide-your-rating-label">YOUR RATING</div>'
        f'    <div class="slide-your-rating">{row["rating"]}<span style="font-size:1.2rem;color:#444">/5</span></div>'
        f'  </div>'
        f'</div>'
    )

# Dots for navigation
dots_html = ""
for i in range(total_slides):
    active_cls = " active" if i == 0 else ""
    dots_html += f'<div class="slide-dot{active_cls}" title="Slide {i+1}"></div>'

# ── Share card content ─────────────────────────────────────────────────────────
sc_dna_segs = ""
sc_pills    = ""
pill_colors = [
    ("rgba(229,9,20,.12)",  "rgba(229,9,20,.4)",  "#ff6666"),
    ("rgba(245,197,24,.1)", "rgba(245,197,24,.35)","#c49010"),
    ("rgba(76,175,80,.1)",  "rgba(76,175,80,.35)", "#4CAF50"),
    ("rgba(33,150,243,.1)", "rgba(33,150,243,.35)","#64b5f6"),
    ("rgba(255,87,34,.1)",  "rgba(255,87,34,.35)", "#ff8a65"),
    ("rgba(233,30,99,.1)",  "rgba(233,30,99,.35)", "#f48fb1"),
    ("rgba(156,39,176,.1)", "rgba(156,39,176,.35)","#ce93d8"),
    ("rgba(0,188,212,.1)",  "rgba(0,188,212,.35)", "#4dd0e1"),
]
for i, (genre_name, genre_score) in enumerate(genres):
    color = genre_colors_list[i] if i < len(genre_colors_list) else "#888"
    flex  = round(genre_score, 3)
    sc_dna_segs += f'<div class="sc-dna-seg" style="background:{color};opacity:{flex};flex:{flex}"></div>'
    if i < len(pill_colors):
        pb, pc, pt = pill_colors[i]
        sc_pills += (
            f'<span class="sc-genre-pill" style="background:{pb};border-color:{pc};color:{pt}">'
            f'{genre_name.upper()}</span>'
        )

top_genre_name = genres[0][0].upper() if genres else "DRAMA"
films_count    = len(full_history)
avg_rating_val = round(full_history["rating"].mean(), 1) if not full_history.empty else 0.0
top_genres_str = " · ".join(g for g, _ in genres[:3]) if genres else ""

# ── Canvas draw data — JS-safe arrays for PNG download ─────────────────────
sc_dna_bar_data = ", ".join(
    f'["{genre_colors_list[i] if i < len(genre_colors_list) else "#888"}", {round(score_val, 3)}]'
    for i, (_, score_val) in enumerate(genres)
)
genre_pill_names = ", ".join(f'"{g[0].upper()}"' for g in genres)

# ── Pre-build controls HTML (avoids nested f-string inside st.markdown) ───────
if total_slides <= 1:
    controls_html = ""
else:
    controls_html = (
        '<div class="slideshow-controls">'
        '<button class="slide-btn" title="Previous">&#8592;</button>'
        f'<div class="slide-dots" id="cwDots">{dots_html}</div>'
        '<button class="slide-btn" title="Next">&#8594;</button>'
        '</div>'
    )

# ── Render with FIXED DOWNLOAD SCRIPT ─────────────────────────────────────────
st.markdown(f"""
<div class="universe-row">
  <!-- LEFT: DNA CARD -->
  <div class="dna-card">
    <div class="dna-card-body">
      <div class="dna-share-badge">DNA</div>
      <div class="dna-logo">◈ CINEWRAP DNA</div>
      <div class="dna-nickname-label">Your archetype</div>
      <div class="dna-nickname">THE <span>{dna_adj.upper()}</span><br>{dna_noun.upper()}</div>
      <div class="dna-helix">
        <div class="dna-strand">{helix_rungs}</div>
      </div>
      <div class="dna-divider"></div>
      <div class="dna-stats-row">
        <div>
          <div class="dna-rarity-score">{score}</div>
          <div class="dna-rarity-sub">RARITY SCORE</div>
        </div>
        <div class="dna-rarity-right">
          <div class="dna-rarity-right-top">TOP {pct_top}%</div>
          <div class="dna-rarity-right-bot">CINEPHILES</div>
        </div>
      </div>
    </div>
    <div class="dna-card-footer">
      <div class="poster-title" style="color:var(--red);font-family:'Bebas Neue',sans-serif;font-size:0.8rem;letter-spacing:2px">YOUR TASTE DNA</div>
      <div class="poster-date">{top_genres_str}</div>
    </div>
  </div>

  <!-- RIGHT: SLIDESHOW -->
  <div class="slideshow-panel" id="cwSlideshowPanel">
    <div class="slideshow-header">
      <div class="slideshow-label">◈ RECENTLY WATCHED</div>
      <div class="slideshow-counter" id="cwSlideCounter">1 / {total_slides}</div>
    </div>
    <div class="slideshow-body" id="cwSlideshowBody">
      {slides_html}
    </div>
    {controls_html}
  </div>
</div>

<!-- SHARE CARD -->
<p class="section-label">◈ SHARE YOUR WRAP</p>
<div class="sc-canvas" id="cwShareCard">
  <div class="sc-header">
    <div class="sc-brand">CINEWRAP 2026</div>
    <div class="sc-year">FILM YEAR</div>
  </div>
  <div class="sc-nickname">THE <span>{dna_adj.upper()}</span> {dna_noun.upper()}</div>
  <div class="sc-type">{top_genres_str}</div>
  <div class="sc-grid">
    <div class="sc-stat">
      <div class="sc-stat-label">TOP GENRE</div>
      <div class="sc-stat-value">{top_genre_name}</div>
    </div>
    <div class="sc-stat">
      <div class="sc-stat-label">RARITY</div>
      <div class="sc-stat-value gold">{score}</div>
    </div>
    <div class="sc-stat">
      <div class="sc-stat-label">FILMS RATED</div>
      <div class="sc-stat-value red">{films_count}</div>
    </div>
    <div class="sc-stat">
      <div class="sc-stat-label">AVG RATING</div>
      <div class="sc-stat-value">&#9733; {avg_rating_val}</div>
    </div>
  </div>
  <div class="sc-dna-row">{sc_dna_segs}</div>
  <div class="sc-genres">{sc_pills}</div>
  <div class="sc-footer">
    <div class="sc-username">@{username}</div>
    <div class="sc-tag">cinewrap.app #MYWRAP2026</div>
  </div>
</div>
<button class="share-btn" id="cwShareBtn">&#8681; DOWNLOAD YOUR WRAP AS PNG</button>

<script>
(function() {{
  // Polyfill for CanvasRenderingContext2D.roundRect
  if (!CanvasRenderingContext2D.prototype.roundRect) {{
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {{
      if (w < 2 * r) r = w / 2;
      if (h < 2 * r) r = h / 2;
      this.moveTo(x+r, y);
      this.lineTo(x+w-r, y);
      this.quadraticCurveTo(x+w, y, x+w, y+r);
      this.lineTo(x+w, y+h-r);
      this.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
      this.lineTo(x+r, y+h);
      this.quadraticCurveTo(x, y+h, x, y+h-r);
      this.lineTo(x, y+r);
      this.quadraticCurveTo(x, y, x+r, y);
      return this;
    }};
  }}

  function cwInit() {{
    var cwCurSlide = 0;
    var cwTotal = {total_slides};

    function cwSlide(idx) {{
      if (cwTotal <= 1) return;
      var slides = document.querySelectorAll('#cwSlideshowBody .slide');
      var dots = document.querySelectorAll('#cwDots .slide-dot');
      cwCurSlide = ((idx % cwTotal) + cwTotal) % cwTotal;
      slides.forEach(function(s, i) {{
        s.classList.toggle('active', i === cwCurSlide);
      }});
      if (dots.length) dots.forEach(function(d, i) {{
        d.classList.toggle('active', i === cwCurSlide);
      }});
      var counter = document.getElementById('cwSlideCounter');
      if (counter) counter.textContent = (cwCurSlide + 1) + ' / ' + cwTotal;
    }}

    var btnPrev = document.querySelector('.slide-btn[title="Previous"]');
    var btnNext = document.querySelector('.slide-btn[title="Next"]');
    if (btnPrev) btnPrev.addEventListener('click', function() {{ cwSlide(cwCurSlide - 1); }});
    if (btnNext) btnNext.addEventListener('click', function() {{ cwSlide(cwCurSlide + 1); }});

    document.querySelectorAll('.slide-dot').forEach(function(dot, i) {{
      dot.addEventListener('click', function() {{ cwSlide(i); }});
    }});

    if (cwTotal > 1) {{
      setInterval(function() {{ cwSlide(cwCurSlide + 1); }}, 4000);
    }}

    // ── FIXED DOWNLOAD FUNCTION ──────────────────────────────────────────────
    function cwDownloadPNG() {{
      var btn = document.getElementById('cwShareBtn');
      if (!btn) return;
      btn.textContent = '⏳ DRAWING…';
      btn.disabled = true;

      var DPR = 2;
      var W = 480;
      var H = 420;
      var canvas = document.createElement('canvas');
      canvas.width = W * DPR;
      canvas.height = H * DPR;
      var ctx = canvas.getContext('2d');
      ctx.scale(DPR, DPR);

      // ---- Background ----
      ctx.fillStyle = '#0f0000';
      ctx.beginPath();
      ctx.roundRect(0, 0, W, H, 14);
      ctx.fill();

      // Red glow
      var grd = ctx.createRadialGradient(W-20, H-20, 0, W-20, H-20, 160);
      grd.addColorStop(0, 'rgba(229,9,20,0.10)');
      grd.addColorStop(1, 'transparent');
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, W, H);

      // Border
      ctx.strokeStyle = '#2a0808';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(0.5, 0.5, W-1, H-1, 14);
      ctx.stroke();

      function txt(str, x, y, font, color, align) {{
        ctx.font = font;
        ctx.fillStyle = color;
        ctx.textAlign = align || 'left';
        ctx.fillText(str, x, y);
      }}

      // Header
      txt('CINEWRAP 2026', 28, 40, '700 10px "Space Mono",monospace', '#E50914');
      txt('FILM YEAR', W-28, 40, '700 10px "Space Mono",monospace', '#333', 'right');

      // Nickname
      txt('THE', 28, 82, '900 28px "Bebas Neue",sans-serif', '#FFFFFF');
      var adjW = ctx.measureText('THE ').width;
      txt('{dna_adj.upper()} ', 28 + adjW, 82, '900 28px "Bebas Neue",sans-serif', '#E50914');
      var adjW2 = ctx.measureText('THE {dna_adj.upper()} ').width;
      txt('{dna_noun.upper()}', 28 + adjW2, 82, '900 28px "Bebas Neue",sans-serif', '#FFFFFF');
      txt('{top_genres_str}', 28, 104, '400 10px "Space Mono",monospace', '#555');

      // Divider
      ctx.strokeStyle = '#1a1a1a';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(28, 120);
      ctx.lineTo(W-28, 120);
      ctx.stroke();

      // Stat boxes
      var stats = [
        ['TOP GENRE', '{top_genre_name}', '#FFFFFF'],
        ['RARITY',    '{score}',          '#F5C518'],
        ['FILMS RATED','{films_count}',   '#E50914'],
        ['AVG RATING', '★ {avg_rating_val}','#FFFFFF'],
      ];
      var bx = 28, by = 136, bw = (W-56-12)/2, bh = 68;
      stats.forEach(function(s, i) {{
        var col = i % 2, row2 = Math.floor(i/2);
        var x = bx + col*(bw+12), y = by + row2*(bh+10);
        ctx.fillStyle = 'rgba(255,255,255,0.03)';
        ctx.strokeStyle = '#1e1e1e';
        ctx.beginPath();
        ctx.roundRect(x, y, bw, bh, 6);
        ctx.fill();
        ctx.stroke();
        txt(s[0], x+10, y+18, '700 8px "Space Mono",monospace', '#444');
        txt(s[1], x+10, y+46, '400 22px "Bebas Neue",sans-serif', s[2]);
      }});

      // DNA color bar
      var genres = [{sc_dna_bar_data}];
      var barY = 310, barH = 5, barX = 28, barMaxW = W-56;
      var total = genres.reduce(function(a,b){{ return a+b[1]; }}, 0) || 1;
      var cx2 = barX;
      genres.forEach(function(g) {{
        var segW = (g[1]/total) * barMaxW;
        ctx.fillStyle = g[0];
        ctx.globalAlpha = g[1];
        ctx.beginPath();
        ctx.roundRect(cx2, barY, segW, barH, 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        cx2 += segW;
      }});

      // Genre pills
      var pillColors = ['#ff6666','#c49010','#4CAF50','#64b5f6','#ff8a65','#f48fb1','#ce93d8','#4dd0e1'];
      var genreNames = [{genre_pill_names}];
      var px2 = 28, py2 = 328;
      ctx.font = '700 9px "Space Mono",monospace';
      genreNames.forEach(function(g, i) {{
        var color = pillColors[i % pillColors.length];
        var tw = ctx.measureText(g).width;
        var pw = tw + 16, ph = 18;
        if (px2 + pw > W - 28) {{ px2 = 28; py2 += 26; }}
        ctx.fillStyle = color.replace(')', ',0.12)').replace('rgb(', 'rgba(');
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.roundRect(px2, py2, pw, ph, 9);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillText(g, px2+8, py2+13);
        px2 += pw + 6;
      }});

      // Footer
      var footerY = H - 40;
      ctx.strokeStyle = '#1a1a1a';
      ctx.beginPath();
      ctx.moveTo(28, footerY);
      ctx.lineTo(W-28, footerY);
      ctx.stroke();
      txt('@{username}', 28, footerY+18, '400 13px "Bebas Neue",sans-serif', '#555');
      txt('cinewrap.app #MYWRAP2026', W-28, footerY+18, '400 9px "Space Mono",monospace', '#333', 'right');

      // ---- DOWNLOAD via toBlob (more reliable) ----
      canvas.toBlob(function(blob) {{
        var link = document.createElement('a');
        link.download = 'cinewrap-{username}-2026.png';
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
        btn.textContent = '✓ SAVED!';
        btn.style.background = '#1a6b1a';
        setTimeout(function() {{
          btn.textContent = '↓ DOWNLOAD YOUR WRAP';
          btn.style.background = '';
          btn.disabled = false;
        }}, 2000);
      }}, 'image/png');
    }}

    var shareBtn = document.getElementById('cwShareBtn');
    if (shareBtn) shareBtn.addEventListener('click', cwDownloadPNG);
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', cwInit);
  }} else {{
    setTimeout(cwInit, 0);
  }}
}})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🍿 RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
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

    cards = [rec_card(row, i + 1, show_posters) for i, (_, row) in enumerate(recs_ranked.iterrows())]
    all_cards = "".join(cards)
    st.markdown(
        '<div class="movie-grid cols-5">' + all_cards + "</div>",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;padding:1.5rem 0 0.5rem;"
    "color:#222;font-size:0.72rem;font-family:DM Sans,sans-serif;letter-spacing:2px'>"
    "CINEWRAP 2026 &nbsp;&middot;&nbsp; CONTENT + COLLABORATIVE + POPULARITY + COUNTRY"
    "</div>",
    unsafe_allow_html=True,
)
