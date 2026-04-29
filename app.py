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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

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
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.2) !important;
}
.stTextInput label { color: #888 !important; font-size:0.8rem !important; }
.stSelectbox [data-baseweb="select"] { background-color: var(--card2) !important; border-color: var(--border) !important; border-radius: 8px !important; }
.stSelectbox [data-baseweb="select"]:focus { border-color: var(--red) !important; }
.stSelectbox label { color: #888 !important; font-size:0.8rem !important; }
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
    font-family: 'DM Sans', sans-serif;
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
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ══════════════════════════════════════════════════════
   UNIVERSE ROW — 50/50 split: DNA card | Slideshow
   ══════════════════════════════════════════════════════ */
.universe-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 28px;
    align-items: start;
}

/* DNA CARD */
.dna-card {
    background: #0f0000;
    border: 1px solid #2a0808;
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 500px;
    position: relative;
    transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
}
.dna-card:hover {
    border-color: var(--red);
    transform: translateY(-4px);
    box-shadow: 0 20px 50px rgba(229,9,20,0.25);
}
.dna-card-body {
    flex: 1;
    padding: 22px 20px 16px;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    min-height: 0;
}
.dna-card-body::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(229,9,20,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.dna-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--red);
    margin-bottom: 8px;
}
.dna-nickname-label {
    font-size: 9px;
    letter-spacing: 2px;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
}
.dna-nickname {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 2px;
    color: var(--white);
    line-height: 1.05;
    margin-bottom: 20px;
}
.dna-nickname span { color: var(--red); }
.dna-helix { flex: 1; overflow: hidden; min-height: 0; }
.dna-strand { display: flex; flex-direction: column; gap: 7px; height: 100%; justify-content: center; }
.dna-rung {
    display: flex;
    align-items: center;
    gap: 5px;
    width: 100%;
    min-width: 0;
    animation: dnafloat 3s ease-in-out infinite;
}
.dna-rung:nth-child(2) { animation-delay: .3s; }
.dna-rung:nth-child(3) { animation-delay: .6s; }
.dna-rung:nth-child(4) { animation-delay: .9s; }
.dna-rung:nth-child(5) { animation-delay: 1.2s; }
.dna-rung:nth-child(6) { animation-delay: 1.5s; }
@keyframes dnafloat { 0%,100%{transform:translateX(0)} 50%{transform:translateX(2px)} }
.dna-dot-l, .dna-dot-r { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dna-bar {
    height: 3px;
    border-radius: 2px;
    flex: 1;
    min-width: 0;
    max-width: 100%;
}
.dna-genre-label {
    font-size: 7px;
    font-family: 'Space Mono', monospace;
    color: #555;
    white-space: nowrap;
    flex-shrink: 0;
    font-weight: 700;
    width: 44px;
    text-align: right;
}
.dna-divider {
    height: 1px;
    background: linear-gradient(90deg, #2a0808, transparent);
    margin: 16px 0;
}
.dna-stats-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.dna-rarity-score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: var(--gold);
    line-height: 1;
}
.dna-rarity-sub {
    font-size: 8px;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-top: 2px;
}
.dna-rarity-right { text-align: right; }
.dna-rarity-right-top {
    font-size: 1.4rem;
    color: #777;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}
.dna-rarity-right-bot {
    font-size: 8px;
    color: #444;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}
.dna-card-footer {
    padding: 14px 24px;
    border-top: 1px solid #1a0808;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.dna-share-badge {
    position: absolute;
    top: 14px; right: 14px;
    background: rgba(229,9,20,0.15);
    border: 1px solid rgba(229,9,20,0.3);
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 8px;
    color: var(--red);
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

/* SLIDESHOW */
.slideshow-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 500px;
    position: relative;
}
.slideshow-panel:hover {
    border-color: #3a3a3a;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.slideshow-header {
    padding: 16px 20px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.slideshow-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 11px;
    letter-spacing: 4px;
    color: #555;
}
.slideshow-counter {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #444;
}
.slideshow-body {
    flex: 1;
    position: relative;
    overflow: hidden;
}
.slide {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    flex-direction: row;
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}
.slide.active {
    opacity: 1;
    pointer-events: auto;
}
.slide-poster {
    width: 45%;
    flex-shrink: 0;
    overflow: hidden;
    background: #111;
}
.slide-poster img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center top;
    display: block;
}
.slide-info {
    flex: 1;
    padding: 24px 20px 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow: hidden;
    min-width: 0;
}
.slide-title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--white);
    line-height: 1.3;
}
.slide-rating { color: var(--gold); font-size: 1rem; }
.slide-date { color: var(--muted); font-size: 0.78rem; }
.slide-your-rating {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--red);
    letter-spacing: 1px;
    line-height: 1;
    margin-top: auto;
}
.slide-your-rating-label {
    font-size: 0.7rem;
    color: #555;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}
.slide-imdb-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--gold);
    width: fit-content;
}
.slideshow-controls {
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
}
.slide-btn {
    background: #1a1a1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #888;
    font-size: 1rem;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
    font-family: monospace;
    flex-shrink: 0;
}
.slide-btn:hover { background: #222; border-color: var(--red); color: var(--red); }
.slide-dots {
    display: flex;
    gap: 6px;
    align-items: center;
}
.slide-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2a2a2a;
    transition: background 0.25s, transform 0.25s;
    cursor: pointer;
}
.slide-dot.active { background: var(--red); transform: scale(1.4); }

/* Single-movie fallback (when only 1 recent movie) */
.single-recent-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 520px;
}
.single-recent-card .poster-img-wrap {
    width: 100%;
    aspect-ratio: 16/9;
    overflow: hidden;
    background: #1a1a1a;
    flex-shrink: 0;
}
.single-recent-card .poster-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* ── Shareable card ── */
.sc-canvas {
    background: #0f0000;
    border-radius: 16px;
    border: 1px solid #2a0808;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
}
.sc-canvas::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(229,9,20,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.sc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.sc-brand { font-family: 'Bebas Neue', sans-serif; font-size: 11px; letter-spacing: 4px; color: var(--red); }
.sc-year  { font-family: 'Bebas Neue', sans-serif; font-size: 11px; letter-spacing: 3px; color: #333; }
.sc-nickname {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 30px;
    letter-spacing: 3px;
    color: var(--white);
    line-height: 1;
    margin-bottom: 4px;
}
.sc-nickname span { color: var(--red); }
.sc-type {
    font-size: 10px;
    color: #555;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 16px;
}
.sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
.sc-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 8px 10px;
}
.sc-stat-label {
    font-size: 8px;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 3px;
}
.sc-stat-value { font-family: 'Bebas Neue', sans-serif; font-size: 18px; color: var(--white); line-height: 1; }
.sc-stat-value.gold { color: var(--gold); }
.sc-stat-value.red  { color: var(--red); }
.sc-dna-row { display: flex; gap: 3px; margin-bottom: 12px; }
.sc-dna-seg { height: 4px; border-radius: 2px; }
.sc-genres  { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 14px; }
.sc-genre-pill {
    font-size: 9px;
    font-family: 'Space Mono', monospace;
    padding: 3px 8px;
    border-radius: 20px;
    font-weight: 700;
    border: 1px solid;
}
.sc-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid #1a1a1a;
}
.sc-username { font-family: 'Bebas Neue', sans-serif; font-size: 13px; letter-spacing: 2px; color: #555; }
.sc-tag { font-size: 9px; color: #333; font-family: 'Space Mono', monospace; }
.share-btn {
    display: block;
    width: 100%;
    margin-top: 12px;
    background: var(--red);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px 22px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 15px;
    letter-spacing: 3px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(229,9,20,0.25);
}
.share-btn:hover  { background: var(--red2); box-shadow: 0 6px 28px rgba(229,9,20,0.4); }
.share-btn:active { transform: scale(0.98); }
.share-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.download-btn {
    display: inline-block;
    margin-top: 10px;
    background: #1a1a1a;
    color: #CCC;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 22px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 14px;
    letter-spacing: 2px;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
}
.download-btn:hover { background: #222; border-color: var(--red); color: var(--red); }

.section-label {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 4px !important;
    color: #555 !important;
    margin-bottom: 10px !important;
}
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
# FIXED SIDEBAR UI: replaced sliders with dropdowns, multiples of 5/10
with st.sidebar:
    st.markdown("### 🎬 CineWrap 2026")
    st.markdown(
        "<p style='color:#555;font-size:0.78rem;margin-top:-8px;'>Hybrid Movie Recommender</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### 🎯 Recommendations")
    # Dropdown instead of slider, multiples of 10: 10,20,30,40,50
    top_n_recs = st.selectbox(
        "How many picks?",
        options=[10, 20, 30, 40, 50],
        index=0,
        help="Number of movie recommendations (multiples of 10)"
    )
    st.markdown("#### 📅 Watch History")
    # Dropdown instead of slider, multiples of 5: 5,10,15,20,25
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

# ── Empty state ────────────────────────────────────────────────────────────────
if not submitted or not user_input.strip():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
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
    .cw-sub{font-family:'DM Sans',sans-serif;font-size:0.74rem;letter-spacing:3px;color:#444;text-transform:uppercase;margin-bottom:2rem;animation:cwSubBreathe 3.5s ease-in-out infinite alternate}
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
      if(reel){for(var i=0;i<24;i++){var h=document.createElement('div');h.className='cw-reel-hole';reel.appendChild(h);var c=document.createElement('div');c.className='cw-reel-cell'+(i%3===1?' cw-lit':'');reel.appendChild(c);}var hx=document.createElement('div');hx.className='cw-reel-hole';reel.appendChild(hx);}
      var heights=['','tall',''];var accents=[false,false,true,false,false,false,true,false,false,false,true,false,false,false,false,true];
      function makePosters(id,count){var row=document.getElementById(id);if(!row)return;for(var i=0;i<count*2;i++){var cls='cw-poster'+(heights[i%3]?' '+heights[i%3]:'')+(accents[i%accents.length]?' accent':'');var delay=((i%count)*0.07).toFixed(2);var barDelay=(parseFloat(delay)+0.15).toFixed(2);var div=document.createElement('div');div.className=cls;div.style.animationDelay=delay+'s';div.innerHTML='<div class="cw-poster-shimmer"></div><div class="cw-poster-inner"><div class="cw-poster-bar" style="animation-delay:'+barDelay+'s"></div><div class="cw-poster-line"></div><div class="cw-poster-line2"></div></div>';row.appendChild(div);}}
      makePosters('cwPosterRow1',14);makePosters('cwPosterRow2',14);
      var ticker=document.getElementById('cwTicker');
      if(ticker){var films=['PARASITE','INCEPTION','SPIRITED AWAY','THE GODFATHER','OLDBOY','MULHOLLAND DRIVE','AMÉLIE','BLADE RUNNER 2049','2001: A SPACE ODYSSEY','GOODFELLAS','PRINCESS MONONOKE','THE DARK KNIGHT','CITY OF GOD','HEAT','FARGO','AKIRA','NOSFERATU','METROPOLIS','WILD STRAWBERRIES','BREATHLESS'];var doubled=films.concat(films);doubled.forEach(function(f,i){var s=document.createElement('span');s.className='cw-t-item';s.textContent=f;ticker.appendChild(s);if(i<doubled.length-1){var d=document.createElement('span');d.className='cw-t-dot';d.textContent='◆';ticker.appendChild(d);}});}
      var stage=document.getElementById('cwStage');var is=document.getElementById('cwIconStage');
      if(stage&&is){function getCenter(){var sr=stage.getBoundingClientRect();var ir=is.getBoundingClientRect();return{x:ir.left-sr.left+50,y:ir.top-sr.top+50};}setInterval(function(){var c=getCenter();var p=document.createElement('div');p.className='cw-particle';var angle=Math.random()*Math.PI*2;var dist=70+Math.random()*100;var dx=(Math.cos(angle)*dist).toFixed(1);var dy=(Math.sin(angle)*dist).toFixed(1);var sz=(2+Math.random()*3).toFixed(1);var dur=(1.6+Math.random()*1.6).toFixed(2);p.style.cssText='width:'+sz+'px;height:'+sz+'px;left:'+(c.x+Math.cos(angle)*10).toFixed(1)+'px;top:'+(c.y+Math.sin(angle)*10).toFixed(1)+'px;--dx:'+dx+'px;--dy:'+dy+'px;animation:cwPFly '+dur+'s ease-out forwards';stage.appendChild(p);setTimeout(function(){if(p.parentNode)p.parentNode.removeChild(p);},parseFloat(dur)*1000+100);},180);
      var dots=[];for(var i=0;i<10;i++){var dot=document.createElement('div');dot.style.cssText='width:4px;height:4px;border-radius:50%;background:#E50914;position:absolute;top:50%;left:50%;margin:-2px;pointer-events:none;';is.appendChild(dot);dots.push({el:dot,offset:(i/10)*Math.PI*2});}
      var frame=0;function animDots(){frame+=0.018;for(var i=0;i<dots.length;i++){var a=dots[i].offset+frame;var x=(Math.cos(a)*40).toFixed(2),y=(Math.sin(a)*40).toFixed(2);dots[i].el.style.transform='translate(calc(-50% + '+x+'px),calc(-50% + '+y+'px))';dots[i].el.style.opacity=(0.3+0.7*((Math.sin(a*2+i)*0.5)+0.5)).toFixed(2);}requestAnimationFrame(animDots);}animDots();}
    })();
    </script>
    """, unsafe_allow_html=True)
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

# ── Render ─────────────────────────────────────────────────────────────────────
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
(function(){{
  /* ── Wait for DOM then wire everything up ── */
  function cwInit() {{

  var cwCurSlide = 0;
  var cwTotal = {total_slides};

  /* ── Slideshow nav ── */
  function cwSlide(idx) {{
    if (cwTotal <= 1) return;
    var slides = document.querySelectorAll('#cwSlideshowBody .slide');
    var dots   = document.querySelectorAll('#cwDots .slide-dot');
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

  /* Wire arrow buttons via event listeners (not onclick attr) */
  var btnPrev = document.querySelector('.slide-btn[title="Previous"]');
  var btnNext = document.querySelector('.slide-btn[title="Next"]');
  if (btnPrev) btnPrev.addEventListener('click', function() {{ cwSlide(cwCurSlide - 1); }});
  if (btnNext) btnNext.addEventListener('click', function() {{ cwSlide(cwCurSlide + 1); }});

  /* Wire dot clicks */
  document.querySelectorAll('.slide-dot').forEach(function(dot, i) {{
    dot.addEventListener('click', function() {{ cwSlide(i); }});
  }});

  /* Auto-advance */
  if (cwTotal > 1) {{
    setInterval(function() {{ cwSlide(cwCurSlide + 1); }}, 4000);
  }}

  /* ── Download PNG — pure canvas draw ── */
  function cwDownloadPNG() {{
    var btn = document.getElementById('cwShareBtn');
    if (!btn) return;
    btn.textContent = '⏳ DRAWING…';
    btn.disabled = true;

    var DPR   = 2;
    var W     = 480;
    var H     = 420;
    var canvas = document.createElement('canvas');
    canvas.width  = W * DPR;
    canvas.height = H * DPR;
    var ctx = canvas.getContext('2d');
    ctx.scale(DPR, DPR);

    /* background */
    ctx.fillStyle = '#0f0000';
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 14);
    ctx.fill();

    /* red glow bottom-right */
    var grd = ctx.createRadialGradient(W-20, H-20, 0, W-20, H-20, 160);
    grd.addColorStop(0, 'rgba(229,9,20,0.10)');
    grd.addColorStop(1, 'transparent');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);

    /* border */
    ctx.strokeStyle = '#2a0808';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, 0.5, W-1, H-1, 14);
    ctx.stroke();

    /* helpers */
    function txt(str, x, y, font, color, align) {{
      ctx.font = font;
      ctx.fillStyle = color;
      ctx.textAlign = align || 'left';
      ctx.fillText(str, x, y);
    }}

    /* CINEWRAP 2026 */
    txt('CINEWRAP 2026', 28, 40, '700 10px "Space Mono",monospace', '#E50914');
    txt('FILM YEAR', W-28, 40, '700 10px "Space Mono",monospace', '#333', 'right');

    /* nickname */
    txt('THE', 28, 82, '900 28px "Bebas Neue",sans-serif', '#FFFFFF');
    var adjW = ctx.measureText('THE ').width;
    txt('{dna_adj.upper()} ', 28 + adjW, 82, '900 28px "Bebas Neue",sans-serif', '#E50914');
    var adjW2 = ctx.measureText('THE {dna_adj.upper()} ').width;
    txt('{dna_noun.upper()}', 28 + adjW2, 82, '900 28px "Bebas Neue",sans-serif', '#FFFFFF');

    /* subtitle */
    txt('{top_genres_str}', 28, 104, '400 10px "Space Mono",monospace', '#555');

    /* divider */
    ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(28, 120); ctx.lineTo(W-28, 120); ctx.stroke();

    /* 4 stat boxes */
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
      ctx.strokeStyle = '#1e1e1e'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(x, y, bw, bh, 6); ctx.fill(); ctx.stroke();
      txt(s[0], x+10, y+18, '700 8px "Space Mono",monospace', '#444');
      txt(s[1], x+10, y+46, '400 22px "Bebas Neue",sans-serif', s[2]);
    }});

    /* DNA colour bar */
    var genres = [{sc_dna_bar_data}];
    var barY = 310, barH = 5, barX = 28, barMaxW = W-56;
    var total = genres.reduce(function(a,b){{return a+b[1];}}, 0) || 1;
    var cx2 = barX;
    genres.forEach(function(g) {{
      var segW = (g[1]/total)*barMaxW;
      ctx.fillStyle = g[0];
      ctx.globalAlpha = g[1];
      ctx.beginPath(); ctx.roundRect(cx2, barY, segW, barH, 2); ctx.fill();
      ctx.globalAlpha = 1;
      cx2 += segW;
    }});

    /* genre pills */
    var pillColors = ['#ff6666','#c49010','#4CAF50','#64b5f6','#ff8a65','#f48fb1','#ce93d8','#4dd0e1'];
    var genreNames = [{genre_pill_names}];
    var px2 = 28, py2 = 328;
    ctx.font = '700 9px "Space Mono",monospace';
    genreNames.forEach(function(g, i) {{
      var color = pillColors[i % pillColors.length];
      var tw = ctx.measureText(g).width;
      var pw = tw + 16, ph = 18;
      if (px2 + pw > W - 28) {{ px2 = 28; py2 += 26; }}
      ctx.fillStyle = color.replace(')', ',0.12)').replace('rgb(','rgba(') || 'rgba(229,9,20,0.12)';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(px2, py2, pw, ph, 9); ctx.fill(); ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(g, px2+8, py2+13);
      px2 += pw + 6;
    }});

    /* footer */
    var footerY = H - 40;
    ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(28, footerY); ctx.lineTo(W-28, footerY); ctx.stroke();
    txt('@{username}', 28, footerY+18, '400 13px "Bebas Neue",sans-serif', '#555');
    txt('cinewrap.app #MYWRAP2026', W-28, footerY+18, '400 9px "Space Mono",monospace', '#333', 'right');

    /* download */
    setTimeout(function() {{
      var link = document.createElement('a');
      link.download = 'cinewrap-{username}-2026.png';
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      btn.textContent = '✓ SAVED!';
      btn.style.background = '#1a6b1a';
      setTimeout(function() {{
        btn.textContent = '↓ DOWNLOAD YOUR WRAP';
        btn.style.background = '';
        btn.style.disabled = false;
        btn.disabled = false;
      }}, 2200);
    }}, 60);
  }} /* end cwDownloadPNG */

  /* Wire download button */
  var shareBtn = document.getElementById('cwShareBtn');
  if (shareBtn) shareBtn.addEventListener('click', cwDownloadPNG);

  }}}} /* end cwInit */

  /* Run after DOM is painted */
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

    cards     = [rec_card(row, i + 1, show_posters) for i, (_, row) in enumerate(recs_ranked.iterrows())]
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
