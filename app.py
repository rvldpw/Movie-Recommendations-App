import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.recommender import RecommenderSystem

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 MovieWrap", layout="wide")

st.title("🎬 MovieWrap — Your Netflix Wrapped")
st.caption("Enter your User ID and get your personal movie wrap + recommendations.")


# ── Load & cache the model ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training recommender…")
def load_system() -> RecommenderSystem:
    df = load_data("data/data_sample.csv")
    model = RecommenderSystem(df)
    model.fit()
    return model


system = load_system()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    top_n_recs = st.slider("Number of recommendations", 5, 20, 10)
    top_n_recent = st.slider("Recent movies to show", 3, 10, 5)
    st.divider()
    st.markdown("**Sample User IDs**")
    sample_ids = system.all_user_ids()[:10]
    st.code(", ".join(str(u) for u in sample_ids))

# ── User ID input ──────────────────────────────────────────────────────────────
user_input = st.text_input("🔍 Enter User ID", placeholder="e.g. 99476")

if not user_input:
    st.info("Enter a User ID in the box above to get started.")
    st.stop()

# Validate input
try:
    user_id = int(user_input.strip())
except ValueError:
    st.error("Please enter a valid integer User ID.")
    st.stop()

if not system.user_exists(user_id):
    st.error(
        f"User **{user_id}** not found in the dataset. "
        f"Try one of the sample IDs in the sidebar."
    )
    st.stop()

# ── Fetch data ─────────────────────────────────────────────────────────────────
with st.spinner("Crunching your movie data…"):
    recent = system.get_recent_activity(user_id, top_n=top_n_recent)
    recs = system.recommend(user_id, top_n=top_n_recs)
    genres = system.get_user_profile(user_id)

# ── Recent Activity ────────────────────────────────────────────────────────────
st.header("🎞️ Your Recent Movie Phase")

if recent.empty:
    st.warning("No recent activity found for this user.")
else:
    cols = st.columns(min(len(recent), 5))
    for col, (_, row) in zip(cols, recent.iterrows()):
        with col:
            stars = "⭐" * int(round(row["rating"]))
            st.metric(label=row["title"], value=f"{row['rating']:.1f}/5")
            st.caption(f"{stars}\n📅 {row['datetime'].date()}")

# ── Recommendations ────────────────────────────────────────────────────────────
st.divider()
st.header("🍿 Next Movies You May Like")

if recs.empty:
    st.warning("Not enough data to generate recommendations.")
else:
    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        pct = int(row["score"] * 100)
        with st.expander(f"#{i} — {row['title']}  ({pct}% match)"):
            st.progress(pct)
            st.caption(
                "Recommended based on your genre taste and what similar users enjoyed."
            )

# ── Taste DNA ──────────────────────────────────────────────────────────────────
st.divider()
st.header("🧬 Your Taste DNA")

if genres:
    genre_df = pd.DataFrame(genres, columns=["Genre", "Score"])
    genre_df = genre_df.set_index("Genre")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(genre_df, use_container_width=True)
    with col2:
        st.dataframe(
            genre_df.style.format({"Score": "{:.1%}"}),
            use_container_width=True,
        )
else:
    st.warning("Could not build a genre profile for this user.")
