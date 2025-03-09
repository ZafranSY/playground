import json  #this is in data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
def load_movie_data(file_path='data.json'):
    """
    Load movie data from JSON file.
    """
    try:
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Successfully loaded {len(data)} items from {file_path}")
            return data
    except FileNotFoundError:
        error_msg = f"The file {file_path} was not found. Please ensure the file exists."
        print(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError:
        error_msg = f"The file {file_path} could not be decoded. Ensure it is a valid JSON file."
        print(error_msg)
        raise ValueError(error_msg)

def prepare_data(movies_data, user_ratings_path=None):
    """
    Prepare and merge movie feature data with user ratings.
    """
    # Convert JSON data to DataFrame
    movies_df = pd.DataFrame(movies_data)

    # Handle user ratings if available
    if user_ratings_path:
        try:
            ratings_df = pd.read_csv(user_ratings_path)
        except FileNotFoundError:
            print(f"Ratings file {user_ratings_path} not found. Proceeding with empty ratings.")
            ratings_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
    else:
        ratings_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

    # Add movie_id if not present (use index)
    if 'movie_id' not in movies_df.columns:
        movies_df['movie_id'] = movies_df.index

    # Create movie feature vectors
    all_genres = set()
    for genres in movies_df['Genre']:
        if isinstance(genres, list):
            all_genres.update(genres)

    # One-hot encode genres
    for genre in all_genres:
        movies_df[f'genre_{genre}'] = movies_df['Genre'].apply(
            lambda x: 1 if genre in x else 0 if isinstance(x, list) else 0
        )

    # Normalize numerical features
    scaler = StandardScaler()
    if 'Mood Intensity' in movies_df.columns and 'Stress Level' in movies_df.columns:
        # Convert stress level to numeric if it's categorical
        if movies_df['Stress Level'].dtype == 'object':
            stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
            movies_df['stress_level_numeric'] = movies_df['Stress Level'].map(stress_map)
        else:
            movies_df['stress_level_numeric'] = movies_df['Stress Level']

        # Scale the numeric features
        numeric_features = movies_df[['Mood Intensity', 'stress_level_numeric']].values
        scaled_features = scaler.fit_transform(numeric_features)
        movies_df['mood_intensity'] = scaled_features[:, 0]
        movies_df['stress_level'] = scaled_features[:, 1]

    # One-hot encode categorical features
    mood_dummies = pd.get_dummies(movies_df['Mood'], prefix='mood')
    emotion_dummies = pd.get_dummies(movies_df['Emotion'], prefix='emotion')

    # Combine all features
    feature_columns = list(mood_dummies.columns) + list(emotion_dummies.columns) + \
                      [col for col in movies_df.columns if col.startswith('genre_')] + \
                      ['mood_intensity', 'stress_level']

    # Create movie features DataFrame
    movies_features = pd.concat([
        movies_df[['movie_id', 'Title']],
        mood_dummies,
        emotion_dummies,
        movies_df[[col for col in movies_df.columns if col.startswith('genre_')]],
        movies_df[['mood_intensity', 'stress_level']]
    ], axis=1)

    # Merge with ratings if available
    if not ratings_df.empty and not movies_features.empty:
        data = pd.merge(ratings_df, movies_features, on='movie_id')
    else:
        data = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'] + list(movies_features.columns))

    return data, movies_features, feature_columns