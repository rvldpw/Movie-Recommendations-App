import streamlit as st
import pandas as pd

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from recommender import RecommenderSystem
from data_loader import load_data

st.set_page_config(
page_title='MovieWrap',
layout='wide'
)

st.title('🎬 Netflix Wrapped')
st.caption(
'Type your User ID and get your personal movie wrap + recommendations'
)


@st.cache_resource
def load_system():

    df=load_data(
        'data/data_sample.csv'
    )

    model=(
        RecommenderSystem(df)
    )

    model.fit()

    return model


system=load_system()


user_id=st.text_input(
'Enter User ID'
)


if user_id:

    user_id=int(user_id)

    recent=(
        system.get_recent_activity(
            user_id,
            top_n=5
        )
    )

    recs=(
        system.recommend(
            user_id,
            top_n=10
        )
    )

    genres=(
        system.get_user_profile(
            user_id
        )
    )


    st.header(
        '🔥 Your Recent Movie Phase'
    )

    for _,r in recent.iterrows():
        st.markdown(
f'''### {r.title}
⭐ Rating: {r.rating}
Watched: {r.datetime.date()}
'''
)


    st.divider()


    st.header(
        '🎯 Next Movies You May Like'
    )

    for _,r in recs.iterrows():
        st.markdown(
f'''### {r.title}
Match Score: {round(r.score,3)}
Recommended because it matches your taste + similar users liked it.
'''
)


    st.divider()

    st.header(
        '📊 Taste DNA'
    )

    genre_df=pd.DataFrame(
        genres,
        columns=[
            'Genre',
            'Score'
        ]
    )

    st.bar_chart(
        genre_df.set_index(
            'Genre'
        )
    )
