import json

def load_movie_data():
    """Load movie data from JSON or return sample data."""
    try:
        with open('data.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return [
            {
                "Title": "The Godfather",
                "Year": 1972,
                "Duration": "175 min",
                "Genre": ["Crime", "Drama"],
                "Emotion": "Intense",
                "Mood": "Gripping",
                "Mood Intensity": 9,
                "Stress Level": "Medium",
                "Best For": "Classic storytelling",
                "Suggested Time to Watch": "Evening"
            }
        ]
