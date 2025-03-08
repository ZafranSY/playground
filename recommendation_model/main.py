from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from data import load_movie_data
from model import create_feature_vectors, get_recommendations

app = FastAPI()

# Load and preprocess movie data
movie_data = load_movie_data()
df = pd.DataFrame(movie_data)
feature_vectors, all_emotions, all_moods, all_genres = create_feature_vectors(df)

class UserInput(BaseModel):
    emotion: str
    mood: str
    mood_intensity: int
    stress_level: str
    genres: list[str]

@app.get("/")
def root():
    return {"message": "Welcome to the Movie Mood Matcher API"}

@app.get("/options")
def get_options():
    """Return available emotions, moods, and genres."""
    return {
        "emotions": all_emotions,
        "moods": all_moods,
        "genres": all_genres
    }

@app.post("/recommend")  # Change from "/recommendations" to "/recommend"
def recommendations(user_input: UserInput):
    """Get movie recommendations based on user input."""
    recs = get_recommendations(
        df, 
        user_input.dict(), 
        feature_vectors, 
        all_emotions, 
        all_moods, 
        all_genres
    )
    return {"recommendations": recs}
