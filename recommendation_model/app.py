from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import pandas as pd
import numpy as np
import traceback
import json
import os

app = FastAPI(title="Movie Recommendation API - Simplified")

# Global store structure initialized with default values
STORE = {
    "movies_df": pd.DataFrame(),
    "feature_vectors": np.array([[1.0, 0.5, 0.6]]),  # Default placeholder
    "all_emotions": ['sad', 'happy', 'intense'],  # Default basic emotions
    "all_moods": ['sad', 'happy', 'uplifting'],    # Default basic moods
    "all_genres": ['drama', 'comedy', 'action'],   # Default basic genres
    "initialized": False
}

class UserInput(BaseModel):
    emotion: str
    mood: str
    mood_intensity: int = 5
    stress_level: str = "Medium"
    genres: List[str] = []

def load_movie_data():
    """Load movie data or create default data if file not found"""
    try:
        # Try multiple possible locations for the data.json file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "data", "data.json"),
            os.path.join(os.path.dirname(__file__), "data.json"),
            "data.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found data file at: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        
        # If no file found, create default data
        print("No data file found. Creating default movie data.")
        return create_default_movie_data()
    except Exception as e:
        print(f"Error loading movie data: {str(e)}")
        return create_default_movie_data()

def create_default_movie_data():
    """Create default movie data when the data file is not available"""
    default_movies = [
        {
            "Title": "Default Drama Movie",
            "Genre": ["Drama"],
            "Emotion": "sad",
            "Mood": "sad",
            "Mood Intensity": 7,
            "Stress Level": "Medium"
        },
        {
            "Title": "Default Comedy Movie",
            "Genre": ["Comedy"],
            "Emotion": "happy",
            "Mood": "happy",
            "Mood Intensity": 8,
            "Stress Level": "Low"
        },
        {
            "Title": "Default Action Movie",
            "Genre": ["Action"],
            "Emotion": "intense",
            "Mood": "uplifting",
            "Mood Intensity": 9,
            "Stress Level": "High"
        }
    ]
    return default_movies

def prepare_data(movie_data):
    """Convert movie data to DataFrame"""
    try:
        df = pd.DataFrame(movie_data)
        return movie_data, df, None
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return [], pd.DataFrame(), None

def create_feature_vectors(df):
    """Create feature vectors from movie data"""
    try:
        # Extract unique values for categoricals
        all_emotions = []
        if 'Emotion' in df.columns:
            all_emotions = df['Emotion'].dropna().unique().tolist()
        
        all_moods = []
        if 'Mood' in df.columns:
            all_moods = df['Mood'].dropna().unique().tolist()
        
        # Extract all unique genres
        all_genres = []
        if 'Genre' in df.columns:
            for genres in df['Genre'].dropna():
                if isinstance(genres, list):
                    all_genres.extend(genres)
            all_genres = list(set(all_genres))
        
        # Create feature vectors (simplified for demonstration)
        n_movies = len(df)
        feature_dim = len(all_emotions) + len(all_moods) + len(all_genres) + 2  # +2 for numerical features
        feature_vectors = np.zeros((n_movies, feature_dim))
        
        for i, (_, row) in enumerate(df.iterrows()):
            vector_idx = 0
            
            # Emotion encoding
            if all_emotions and 'Emotion' in df.columns:
                emotion = row.get('Emotion')
                if emotion:
                    emotion_idx = all_emotions.index(emotion) if emotion in all_emotions else -1
                    if emotion_idx >= 0:
                        feature_vectors[i, emotion_idx] = 1
                vector_idx += len(all_emotions)
            
            # Mood encoding
            if all_moods and 'Mood' in df.columns:
                mood = row.get('Mood')
                if mood:
                    mood_idx = all_moods.index(mood) if mood in all_moods else -1
                    if mood_idx >= 0:
                        feature_vectors[i, vector_idx + mood_idx] = 1
                vector_idx += len(all_moods)
            
            # Genre encoding
            if all_genres and 'Genre' in df.columns:
                genres = row.get('Genre', [])
                if isinstance(genres, list):
                    for genre in genres:
                        genre_idx = all_genres.index(genre) if genre in all_genres else -1
                        if genre_idx >= 0:
                            feature_vectors[i, vector_idx + genre_idx] = 1
                vector_idx += len(all_genres)
            
            # Numerical features
            # Mood Intensity
            mood_intensity = row.get('Mood Intensity', 5)
            feature_vectors[i, vector_idx] = mood_intensity / 10  # Normalize to 0-1
            vector_idx += 1
            
            # Stress Level
            stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
            stress_level = row.get('Stress Level', 'Medium')
            feature_vectors[i, vector_idx] = stress_map.get(stress_level, 0.6)
        
        print(f"Created feature vectors with shape: {feature_vectors.shape}")
        return feature_vectors, all_emotions, all_moods, all_genres
    except Exception as e:
        print(f"Error creating feature vectors: {str(e)}")
        # Return simple defaults if feature creation fails
        return np.ones((len(df), 3)), ['default'], ['default'], ['default']

def get_recommendations(df, user_input, feature_vectors, all_emotions, all_moods, all_genres, top_n=5):
    """Get movie recommendations based on user input"""
    try:
        print(f"Calculating recommendations for: {user_input}")
        print(f"Available emotions: {all_emotions}")
        print(f"Available moods: {all_moods}")
        print(f"Available genres: {all_genres}")
        
        # First try filtering based on exact matches
        filtered = df.copy()
        
        # Filter by emotion if present
        if 'Emotion' in df.columns and user_input['emotion']:
            emotion_match = filtered['Emotion'].str.lower() == user_input['emotion'].lower()
            if emotion_match.any():
                filtered = filtered[emotion_match]
        
        # Filter by mood if present
        if 'Mood' in df.columns and user_input['mood']:
            mood_match = filtered['Mood'].str.lower() == user_input['mood'].lower()
            if mood_match.any():
                filtered = filtered[mood_match]
        
        # Filter by genres if present
        if 'Genre' in df.columns and user_input['genres']:
            genre_filter = filtered.apply(
                lambda row: any(g.lower() in [genre.lower() for genre in row['Genre']] 
                               for g in user_input['genres']) if isinstance(row['Genre'], list) else False,
                axis=1
            )
            if genre_filter.any():
                filtered = filtered[genre_filter]
        
        # If we have matching movies, return them
        if not filtered.empty:
            print(f"Found {len(filtered)} matching movies through direct filtering")
            results = filtered.head(top_n).copy()
            results['Score'] = np.random.uniform(4.0, 5.0, size=len(results))  # High scores for direct matches
            return results.replace({np.nan: None}).to_dict('records')
        
        # If no direct matches, use feature vectors to find similar movies
        print("No direct matches found. Using vector similarity...")
        
        # Create user feature vector
        user_vector = np.zeros(feature_vectors.shape[1])
        vector_idx = 0
        
        # Emotion encoding
        if all_emotions:
            for i, emotion in enumerate(all_emotions):
                if user_input['emotion'].lower() == emotion.lower():
                    user_vector[i] = 1
                    break
            vector_idx += len(all_emotions)
        
        # Mood encoding
        if all_moods:
            for i, mood in enumerate(all_moods):
                if user_input['mood'].lower() == mood.lower():
                    user_vector[vector_idx + i] = 1
                    break
            vector_idx += len(all_moods)
        
        # Genre encoding
        if all_genres:
            for i, genre in enumerate(all_genres):
                if any(g.lower() == genre.lower() for g in user_input['genres']):
                    user_vector[vector_idx + i] = 1
            vector_idx += len(all_genres)
        
        # Numerical features
        user_vector[vector_idx] = user_input['mood_intensity'] / 10
        vector_idx += 1
        
        stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        user_vector[vector_idx] = stress_map.get(user_input['stress_level'], 0.6)
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            similarities = cosine_similarity([user_vector], feature_vectors)[0]
            
            # Get top matches
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            # Create recommendation list
            results = df.iloc[top_indices].copy()
            results['Score'] = similarities[top_indices] * 5  # Scale to 0-5
            
            print(f"Found {len(results)} recommendations through vector similarity")
            return results.replace({np.nan: None}).to_dict('records')
        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            
        # If all else fails, return random recommendations
        print("Falling back to random recommendations")
        results = df.sample(min(top_n, len(df))).copy()
        results['Score'] = np.random.uniform(3.0, 4.5, size=len(results))
        return results.replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        print(traceback.format_exc())
        
        # Absolute fallback: return default recommendation
        return [{"Title": "Default Movie", "Score": np.random.uniform(3.5, 5.0)}]

@app.on_event("startup")
async def startup_event():
    """Initialize data on application startup"""
    try:
        # Load movie data
        print("Loading movie data...")
        movie_data = load_movie_data()
        print(f"Data loaded: {len(movie_data)} movies")
        
        # Prepare data
        _, temp_df, _ = prepare_data(movie_data)
        
        # Create feature vectors
        if not temp_df.empty:
            temp_vectors, temp_emotions, temp_moods, temp_genres = create_feature_vectors(temp_df)
            
            # Update global store
            STORE["movies_df"] = temp_df
            STORE["feature_vectors"] = temp_vectors
            STORE["all_emotions"] = temp_emotions
            STORE["all_moods"] = temp_moods
            STORE["all_genres"] = temp_genres
            STORE["initialized"] = True
            
            print("Data initialization successful!")
            print(f"Feature vectors shape: {temp_vectors.shape}")
            print(f"Available emotions: {temp_emotions}")
            print(f"Available moods: {temp_moods}")
            print(f"Available genres: {temp_genres}")
        else:
            print("WARNING: No movie data was loaded.")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        
        # Create some default data if initialization fails
        default_movies = create_default_movie_data()
        default_df = pd.DataFrame(default_movies)
        default_vectors, default_emotions, default_moods, default_genres = create_feature_vectors(default_df)
        
        STORE["movies_df"] = default_df
        STORE["feature_vectors"] = default_vectors
        STORE["all_emotions"] = default_emotions
        STORE["all_moods"] = default_moods
        STORE["all_genres"] = default_genres
        STORE["initialized"] = True

@app.post("/recommend")
def recommend_movies(user_input: UserInput, limit: int = Query(5, gt=0, le=20)):
    """Get movie recommendations based on user input"""
    try:
        # Convert model to dict
        user_dict = user_input.model_dump()
        print(f"Received user input: {user_dict}")
        
        # Check if data is initialized
        if not STORE["initialized"] or STORE["movies_df"].empty:
            print("WARNING: Data not initialized. Creating default recommendations.")
            return {"recommendations": [{"Title": "Default Movie", "Score": 4.5}]}
        
        # Get recommendations using the global store
        recommendations = get_recommendations(
            STORE["movies_df"],
            user_dict,
            STORE["feature_vectors"],
            STORE["all_emotions"],
            STORE["all_moods"],
            STORE["all_genres"],
            top_n=limit
        )
        
        print(f"Returning {len(recommendations)} recommendations")
        return {"recommendations": recommendations}
    except Exception as e:
        error_msg = str(e)
        print(f"Error in recommendation endpoint: {error_msg}")
        traceback.print_exc()
        return {"recommendations": [{"Title": "Error occurred", "Score": 0, "Error": error_msg}]}

@app.get("/info")
def get_info():
    """Get information about the movie database"""
    try:
        movie_count = len(STORE["movies_df"]) if not STORE["movies_df"].empty else 0
        
        return {
            "movie_count": movie_count,
            "emotions": STORE["all_emotions"],
            "moods": STORE["all_moods"],
            "genres": STORE["all_genres"],
            "feature_vector_shape": STORE["feature_vectors"].shape.tolist() if hasattr(STORE["feature_vectors"], "shape") else None,
            "initialized": STORE["initialized"]
        }
    except Exception as e:
        error_msg = str(e)
        print(f"Error in info endpoint: {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)