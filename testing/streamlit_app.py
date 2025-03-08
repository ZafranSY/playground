import streamlit as st
import pandas as pd
from data import load_movie_data
from model import create_feature_vectors, get_recommendations

# Set Streamlit page config
st.set_page_config(
    page_title="Movie Mood Matcher",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .movie-card {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .movie-title {
        color: #f1c40f;
        font-size: 24px;
        margin-bottom: 5px;
    }
    .movie-info {
        color: #ecf0f1;
        margin-bottom: 5px;
    }
    .match-score {
        font-size: 18px;
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Load and preprocess data
    movie_data = load_movie_data()
    df = pd.DataFrame(movie_data)
    feature_vectors, all_emotions, all_moods, all_genres = create_feature_vectors(df)
    
    # User input
    st.header("Find the Perfect Movie for Your Mood 🎥")
    user_input = {
        'emotion': st.selectbox("What's your current emotion?", all_emotions),
        'mood': st.selectbox("Select a mood:", all_moods),
        'mood_intensity': st.slider("Mood Intensity (1-10):", 1, 10, 5),
        'stress_level': st.selectbox("Stress Level:", ['Low', 'Medium', 'High']),
        'genres': st.multiselect("Preferred Genres:", all_genres)
    }
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(df, user_input, feature_vectors, all_emotions, all_moods, all_genres)
        for rec in recommendations:
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{rec['title']}</div>
                <div class="movie-info">Score: {rec['score']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
