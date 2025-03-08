import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Set page config
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
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Sample data - pre-populated with movies and their attributes
def load_movie_data():
    # Try to load from data.json if it exists
    try:
        with open('data.json', 'r') as file:
            return json.load(file)
    except:
        # Return sample data if file doesn't exist
        return [
            {
                "Title": "The Shawshank Redemption",
                "Year": 1994,
                "Duration": "142 min",
                "Genre": ["Drama"],
                "Emotion": "Hopeful",
                "Mood": "Inspirational",
                "Mood Intensity": 8,
                "Stress Level": "High",
                "Best For": "Feeling trapped or stuck",
                "Suggested Time to Watch": "Evening"
            },
            {
                "Title": "Forrest Gump",
                "Year": 1994,
                "Duration": "142 min",
                "Genre": ["Drama", "Romance"],
                "Emotion": "Emotional",
                "Mood": "Nostalgic",
                "Mood Intensity": 7,
                "Stress Level": "Medium",
                "Best For": "Reflecting on life",
                "Suggested Time to Watch": "Weekend"
            },
            {
                "Title": "The Pursuit of Happyness",
                "Year": 2006,
                "Duration": "117 min",
                "Genre": ["Biography", "Drama"],
                "Emotion": "Inspired",
                "Mood": "Motivational",
                "Mood Intensity": 9,
                "Stress Level": "High",
                "Best For": "Financial struggles",
                "Suggested Time to Watch": "Morning"
            },
            {
                "Title": "Soul",
                "Year": 2020,
                "Duration": "100 min",
                "Genre": ["Animation", "Comedy", "Drama"],
                "Emotion": "Thoughtful",
                "Mood": "Contemplative",
                "Mood Intensity": 6,
                "Stress Level": "Medium",
                "Best For": "Finding purpose",
                "Suggested Time to Watch": "Afternoon"
            },
            {
                "Title": "Inside Out",
                "Year": 2015,
                "Duration": "95 min",
                "Genre": ["Animation", "Comedy", "Drama"],
                "Emotion": "Mixed",
                "Mood": "Emotional",
                "Mood Intensity": 8,
                "Stress Level": "Medium",
                "Best For": "Understanding emotions",
                "Suggested Time to Watch": "Any time"
            },
            {
                "Title": "The Hangover",
                "Year": 2009,
                "Duration": "100 min",
                "Genre": ["Comedy"],
                "Emotion": "Amused",
                "Mood": "Silly",
                "Mood Intensity": 7,
                "Stress Level": "Low",
                "Best For": "Need to laugh",
                "Suggested Time to Watch": "Night"
            },
            {
                "Title": "Pulp Fiction",
                "Year": 1994,
                "Duration": "154 min",
                "Genre": ["Crime", "Drama"],
                "Emotion": "Intrigued",
                "Mood": "Edgy",
                "Mood Intensity": 8,
                "Stress Level": "Medium",
                "Best For": "Escapism",
                "Suggested Time to Watch": "Night"
            },
            {
                "Title": "The Notebook",
                "Year": 2004,
                "Duration": "123 min",
                "Genre": ["Drama", "Romance"],
                "Emotion": "Romantic",
                "Mood": "Sentimental",
                "Mood Intensity": 9,
                "Stress Level": "Low",
                "Best For": "Heartbreak",
                "Suggested Time to Watch": "Evening"
            },
            {
                "Title": "The Dark Knight",
                "Year": 2008,
                "Duration": "152 min",
                "Genre": ["Action", "Crime", "Drama"],
                "Emotion": "Thrilled",
                "Mood": "Intense",
                "Mood Intensity": 9,
                "Stress Level": "Medium",
                "Best For": "Excitement",
                "Suggested Time to Watch": "Night"
            },
            {
                "Title": "Spirited Away",
                "Year": 2001,
                "Duration": "125 min",
                "Genre": ["Animation", "Adventure", "Family"],
                "Emotion": "Wonderment",
                "Mood": "Magical",
                "Mood Intensity": 7,
                "Stress Level": "Low",
                "Best For": "Escaping reality",
                "Suggested Time to Watch": "Evening"
            },
            {
                "Title": "The Exorcist",
                "Year": 1973,
                "Duration": "122 min",
                "Genre": ["Horror"],
                "Emotion": "Frightened",
                "Mood": "Tense",
                "Mood Intensity": 10,
                "Stress Level": "High",
                "Best For": "Thrill seekers",
                "Suggested Time to Watch": "Night"
            },
            {
                "Title": "Good Will Hunting",
                "Year": 1997,
                "Duration": "126 min",
                "Genre": ["Drama", "Romance"],
                "Emotion": "Reflective",
                "Mood": "Thoughtful",
                "Mood Intensity": 7,
                "Stress Level": "Medium",
                "Best For": "Personal growth",
                "Suggested Time to Watch": "Evening"
            }
        ]

# Create vectors for matching
def create_feature_vectors(df):
    # Get all unique emotions, moods, and genres
    all_emotions = df['Emotion'].unique().tolist()
    all_moods = df['Mood'].unique().tolist()
    
    # Create a flat list of all genres
    all_genres = []
    for genres in df['Genre']:
        all_genres.extend(genres)
    all_genres = list(set(all_genres))
    
    # Create vectors for each movie
    feature_vectors = []
    for _, row in df.iterrows():
        # Initialize vector with zeros
        vector = []
        
        # One-hot encode emotion
        for emotion in all_emotions:
            vector.append(1 if row['Emotion'] == emotion else 0)
        
        # One-hot encode mood
        for mood in all_moods:
            vector.append(1 if row['Mood'] == mood else 0)
        
        # One-hot encode genres
        for genre in all_genres:
            vector.append(1 if genre in row['Genre'] else 0)
        
        # Add mood intensity (normalized)
        vector.append(row['Mood Intensity'] / 10 if 'Mood Intensity' in row else 0.5)
        
        # Add stress level as numeric
        stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        vector.append(stress_map.get(row.get('Stress Level', 'Medium'), 0.6))
        
        feature_vectors.append(vector)
    
    return np.array(feature_vectors), all_emotions, all_moods, all_genres

# Find movie recommendations based on user input
def get_recommendations(df, user_input, feature_vectors, all_emotions, all_moods, all_genres, top_n=3):
    # Create user vector
    user_vector = []
    
    # One-hot encode emotion
    for emotion in all_emotions:
        user_vector.append(1 if user_input['emotion'] == emotion else 0)
    
    # One-hot encode mood
    for mood in all_moods:
        user_vector.append(1 if user_input['mood'] == mood else 0)
    
    # One-hot encode genres if the user specified any
    user_genres = user_input.get('genres', [])
    for genre in all_genres:
        user_vector.append(1 if genre in user_genres else 0)
    
    # Add mood intensity (normalized)
    user_vector.append(user_input.get('mood_intensity', 5) / 10)
    
    # Add stress level
    stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
    user_vector.append(stress_map.get(user_input.get('stress_level', 'Medium'), 0.6))
    
    # Reshape for similarity calculation
    user_vector = np.array(user_vector).reshape(1, -1)
    
    # Calculate similarity scores
    similarity = cosine_similarity(user_vector, feature_vectors)[0]
    
    # Get indices of top recommendations
    top_indices = similarity.argsort()[-top_n:][::-1]
    
    # Create recommendation list
    recommendations = []
    for idx in top_indices:
        movie = df.iloc[idx].to_dict()
        recommendations.append({
            'title': movie['Title'],
            'score': float(similarity[idx] * 100),  # Convert to percentage
            'info': movie
        })
    
    return recommendations

# Main app
def main():
    # Load data
    movies = load_movie_data()
    
    # Convert to DataFrame
    df = pd.DataFrame(movies)
    
    # Create feature vectors for similarity matching
    feature_vectors, all_emotions, all_moods, all_genres = create_feature_vectors(df)
    
    # Title
    st.title("🎬 Movie Mood Matcher")
    st.markdown("Find the perfect movie for your current mood without any training required!")
    
    # Create a clean user input form
    with st.container():
        st.subheader("How are you feeling today?")
        
        # Create two columns for a cleaner layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Primary emotion
            selected_emotion = st.selectbox(
                "Current Emotion",
                options=sorted(all_emotions),
                help="How would you describe your primary emotion right now?"
            )
            
            # Mood intensity
            mood_intensity = st.slider(
                "Emotion Intensity",
                min_value=1,
                max_value=10,
                value=7,
                help="How strongly are you feeling this emotion? (1 = barely, 10 = overwhelmingly)"
            )
        
        with col2:
            # Current mood
            selected_mood = st.selectbox(
                "Current Mood",
                options=sorted(all_moods),
                help="What's your overall mood or atmosphere you're seeking?"
            )
            
            # Stress level
            stress_level = st.selectbox(
                "Current Stress Level",
                options=["Low", "Medium", "High"],
                index=1,
                help="How would you describe your current stress level?"
            )
        
        # Genre preferences (optional)
        genre_preferences = st.multiselect(
            "Any specific genres you're in the mood for? (Optional)",
            options=sorted(all_genres),
            help="Leave empty if you're open to any genre"
        )
        
        # Reason for mood (optional text input)
        reason = st.text_area(
            "What's causing your current mood? (Optional)",
            placeholder="e.g., Had a stressful day at work, just broke up, celebrating a promotion...",
            help="This helps us find more tailored recommendations"
        )
        
        # Get recommendations button
        if st.button("Find My Perfect Movie", type="primary", use_container_width=True):
            # Collect user input
            user_input = {
                'emotion': selected_emotion,
                'mood': selected_mood,
                'mood_intensity': mood_intensity,
                'stress_level': stress_level,
                'genres': genre_preferences,
                'reason': reason
            }
            
            # Get recommendations
            with st.spinner("Finding the perfect movies for you..."):
                recommendations = get_recommendations(
                    df, 
                    user_input, 
                    feature_vectors, 
                    all_emotions, 
                    all_moods, 
                    all_genres
                )
            
            # Display recommendations
            if recommendations:
                st.subheader("📽️ Your Personalized Movie Recommendations")
                st.markdown("Based on your current mood and preferences:")
                
                for i, rec in enumerate(recommendations):
                    with st.container():
                        # Create a movie card with custom HTML
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="header-container">
                                <div class="movie-title">{i+1}. {rec['info']['Title']} ({rec['info']['Year']})</div>
                                <div class="match-score">Match: {rec['score']:.1f}%</div>
                            </div>
                            <div class="movie-info"><b>Genre:</b> {', '.join(rec['info']['Genre'])}</div>
                            <div class="movie-info"><b>Duration:</b> {rec['info']['Duration']}</div>
                            <div class="movie-info"><b>Emotion:</b> {rec['info']['Emotion']}</div>
                            <div class="movie-info"><b>Mood:</b> {rec['info']['Mood']}</div>
                            <div class="movie-info"><b>Best For:</b> {rec['info'].get('Best For', 'Any occasion')}</div>
                            <div class="movie-info"><b>Best Time to Watch:</b> {rec['info'].get('Suggested Time to Watch', 'Anytime')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Explain the recommendation
                st.info(f"""
                Why these recommendations? You're feeling **{selected_emotion.lower()}** with a mood that's **{selected_mood.lower()}** 
                at intensity level **{mood_intensity}/10** with a **{stress_level.lower()}** stress level. 
                {'Your preference for ' + ', '.join(genre_preferences) + ' was also considered.' if genre_preferences else ''}
                """)
            else:
                st.error("No recommendations found. Try different preferences.")

if __name__ == "__main__":
    main()