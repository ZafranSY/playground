# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def create_feature_vectors(df):
#     """Create feature vectors for movie attributes."""
#     all_emotions = df['Emotion'].unique().tolist()
#     all_moods = df['Mood'].unique().tolist()

#     all_genres = []
#     for genres in df['Genre']:
#         all_genres.extend(genres)
#     all_genres = list(set(all_genres))

#     feature_vectors = []
#     for _, row in df.iterrows():
#         vector = []
#         vector += [1 if row['Emotion'] == emotion else 0 for emotion in all_emotions]
#         vector += [1 if row['Mood'] == mood else 0 for mood in all_moods]
#         vector += [1 if genre in row['Genre'] else 0 for genre in all_genres]
#         vector.append(row['Mood Intensity'] / 10 if 'Mood Intensity' in row else 0.5)
#         stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
#         vector.append(stress_map.get(row.get('Stress Level', 'Medium'), 0.6))
#         feature_vectors.append(vector)

#     return np.array(feature_vectors), all_emotions, all_moods, all_genres

# def get_recommendations(df, user_input, feature_vectors, all_emotions, all_moods, all_genres, top_n=3):
#     """Find movie recommendations based on user preferences."""
#     user_vector = []
#     user_vector += [1 if user_input['emotion'] == emotion else 0 for emotion in all_emotions]
#     user_vector += [1 if user_input['mood'] == mood else 0 for mood in all_moods]
#     user_vector += [1 if genre in user_input.get('genres', []) else 0 for genre in all_genres]
#     user_vector.append(user_input.get('mood_intensity', 5) / 10)
#     stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
#     user_vector.append(stress_map.get(user_input.get('stress_level', 'Medium'), 0.6))
#     user_vector = np.array(user_vector).reshape(1, -1)

#     similarity = cosine_similarity(user_vector, feature_vectors)[0]
#     top_indices = similarity.argsort()[-top_n:][::-1]

#     recommendations = []
#     for idx in top_indices:
#         movie = df.iloc[idx].to_dict()
#         recommendations.append({
#             'title': movie['Title'],
#             'score': float(similarity[idx] * 100),
#             'info': movie
#         })

#     return recommendations    import numpy as np thi is model.py
import numpy as np
import pandas as pd  # Add pandas import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
def create_feature_vectors(df):
    """
    Create feature vectors using the movies DataFrame
    Returns the feature vectors and lists of emotions, moods, and genres
    """
    # Debug print to check dataframe structure
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame sample: {df.head(1).to_dict('records')}")
    
    # Extract unique emotions, moods, and genres (case-insensitive)
    all_emotions = []
    if 'Emotion' in df.columns:
        all_emotions = df['Emotion'].dropna().unique().tolist()
    elif 'emotion' in df.columns:  # Try lowercase version
        all_emotions = df['emotion'].dropna().unique().tolist()
    
    all_moods = []
    if 'Mood' in df.columns:
        all_moods = df['Mood'].dropna().unique().tolist()
    elif 'mood' in df.columns:  # Try lowercase version
        all_moods = df['mood'].dropna().unique().tolist()
    
    # Extract all unique genres from the list of genres for each movie
    all_genres = []
    genre_col = None
    if 'Genre' in df.columns:
        genre_col = 'Genre'
    elif 'genre' in df.columns:
        genre_col = 'genre'
    elif any(col.startswith('genre_') for col in df.columns):
        # If we have one-hot encoded genres already
        genre_cols = [col for col in df.columns if col.startswith('genre_')]
        all_genres = [col.replace('genre_', '') for col in genre_cols]
    
    if genre_col:
        for genres in df[genre_col].dropna():
            if isinstance(genres, list):
                all_genres.extend(genres)
            elif isinstance(genres, str):
                # Handle case where genres might be a comma-separated string
                all_genres.extend([g.strip() for g in genres.split(',')])
        all_genres = list(set(all_genres))
    
    print(f"Extracted emotions: {all_emotions}")
    print(f"Extracted moods: {all_moods}")
    print(f"Extracted genres: {all_genres}")
    
    # If we have no features, create at least one dummy feature
    if not all_emotions and not all_moods and not all_genres:
        print("WARNING: No features found in data. Creating a dummy feature.")
        all_emotions = ['default']
        df['Emotion'] = 'default'
    
    # Create feature vectors for each movie
    feature_vectors = []
    for _, row in df.iterrows():
        vector = []
        
        # Add emotion one-hot encoding
        if all_emotions:
            emotion = row.get('Emotion', row.get('emotion', all_emotions[0]))
            vector += [1 if emotion == e else 0 for e in all_emotions]
        
        # Add mood one-hot encoding
        if all_moods:
            mood = row.get('Mood', row.get('mood', all_moods[0]))
            vector += [1 if mood == m else 0 for m in all_moods]
        
        # Add genre one-hot encoding
        if all_genres and genre_col:
            genres = row.get(genre_col, [])
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',')]
            elif not isinstance(genres, list):
                genres = []
            vector += [1 if g in genres else 0 for g in all_genres]
        elif all_genres:
            # Check for one-hot encoded genres
            vector += [row.get(f'genre_{g}', 0) for g in all_genres]
        
        # Add numerical features
        mood_intensity = 0.5  # Default value
        if 'Mood Intensity' in df.columns and not pd.isna(row.get('Mood Intensity')):
            mood_intensity = row['Mood Intensity'] / 10
        elif 'mood_intensity' in df.columns and not pd.isna(row.get('mood_intensity')):
            mood_intensity = row['mood_intensity'] / 10
        vector.append(mood_intensity)
        
        stress_level = 0.6  # Default medium
        stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        
        if 'Stress Level' in df.columns:
            sl = row.get('Stress Level')
            if isinstance(sl, str):
                stress_level = stress_map.get(sl, 0.6)
            elif not pd.isna(sl):
                stress_level = sl
        elif 'stress_level' in df.columns:
            sl = row.get('stress_level')
            if isinstance(sl, str):
                stress_level = stress_map.get(sl, 0.6)
            elif not pd.isna(sl):
                stress_level = sl
        
        vector.append(stress_level)
        
        feature_vectors.append(vector)
    
    # Ensure we have at least one feature
    if not feature_vectors or len(feature_vectors[0]) == 0:
        print("WARNING: Generated empty feature vectors. Adding dummy feature.")
        feature_vectors = [[1] for _ in range(len(df))]
    
    print(f"Feature vector shape: {len(feature_vectors)} x {len(feature_vectors[0])}")
    return np.array(feature_vectors), all_emotions, all_moods, all_genres

def build_content_based_model(movies_features):
    """
    Build a content-based recommendation model using cosine similarity
    """
    # Extract feature matrix (excluding movie_id and Title columns)
    feature_matrix = movies_features.drop(['movie_id', 'Title'], axis=1, errors='ignore').values
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Create a model using K-nearest neighbors
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(feature_matrix)
    
    return model, similarity_matrix, feature_matrix

def build_collaborative_model(data, num_users, num_movies, embedding_size=50):
    """
    Build a neural network-based collaborative filtering model
    """
    # User input
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(num_users + 1, embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_users')(user_embedding)
    
    # Movie input
    movie_input = Input(shape=(1,), name='movie_input')
    movie_embedding = Embedding(num_movies + 1, embedding_size, name='movie_embedding')(movie_input)
    movie_vec = Flatten(name='flatten_movies')(movie_embedding)
    
    # Feature input (for movie features)
    feature_cols = len(data.columns) - 3 if len(data.columns) > 3 else 1  # Excluding user_id, movie_id, rating
    feature_input = Input(shape=(feature_cols,), name='feature_input')
    feature_dense = Dense(100, activation='relu')(feature_input)
    
    # Concatenate layers
    concat = Concatenate()([user_vec, movie_vec, feature_dense])
    
    # Dense layers
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1)(dense2)
    
    # Create model
    model = Model(
        inputs=[user_input, movie_input, feature_input],
        outputs=output
    )
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model

class HybridRecommender:
    def __init__(self, content_model, collaborative_model, movies_features, similarity_matrix, feature_matrix, all_emotions, all_moods, all_genres):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.movies_features = movies_features
        self.similarity_matrix = similarity_matrix
        self.feature_matrix = feature_matrix
        self.all_emotions = all_emotions
        self.all_moods = all_moods
        self.all_genres = all_genres
        
    def get_content_recommendations(self, movie_idx, n=10):
        """Get content-based recommendations"""
        # Get similarity scores
        movie_similarities = self.similarity_matrix[movie_idx]
        
        # Get top similar movies
        similar_movies_indices = movie_similarities.argsort()[-n-1:-1][::-1]
        similar_movies = self.movies_features.iloc[similar_movies_indices]
        
        return similar_movies
    
    def get_user_recommendations(self, user_id, user_features, n=10):
        """
        Get personalized recommendations for a user
        combining collaborative and content-based filtering
        """
        # Create user feature vector
        user_vector = []
        
        # Add emotion one-hot encoding
        user_vector += [1 if user_features['emotion'] == emotion else 0 for emotion in self.all_emotions]
        
        # Add mood one-hot encoding
        user_vector += [1 if user_features['mood'] == mood else 0 for mood in self.all_moods]
        
        # Add genre one-hot encoding
        user_vector += [1 if genre in user_features.get('genres', []) else 0 for genre in self.all_genres]
        
        # Add numerical features
        user_vector.append(user_features.get('mood_intensity', 5) / 10)  # Normalize to 0-1 scale
        
        stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        user_vector.append(stress_map.get(user_features.get('stress_level', 'Medium'), 0.6))
        
        # Get content-based recommendations
        user_vector_np = np.array(user_vector).reshape(1, -1)
        distances, indices = self.content_model.kneighbors(user_vector_np, n_neighbors=min(n, len(self.movies_features)))
        content_recommendations = self.movies_features.iloc[indices[0]]
        
        # Get collaborative recommendations if user exists in training data
        # (simplified here - in practice would use the collaborative model)
        
        # Combine recommendations (simplified approach)
        final_recommendations = content_recommendations
        
        return final_recommendations

def get_recommendations(df, user_preferences, feature_vectors, all_emotions, all_moods, all_genres, top_n=5):
    """Find movie recommendations based on user preferences."""
    # Check if feature vectors are empty
    if feature_vectors.size == 0 or feature_vectors.shape[1] == 0:
        print("WARNING: Feature vectors are empty. Cannot compute recommendations.")
        return [{"title": "No recommendations available", "score": 0, "info": {}}]
    
    user_vector = []
    
    # Add emotion one-hot encoding
    if all_emotions:
        user_emotion = user_preferences.get('emotion', all_emotions[0])
        vector_part = [1 if user_emotion.lower() == emotion.lower() else 0 for emotion in all_emotions]
        user_vector += vector_part
    
    # Add mood one-hot encoding
    if all_moods:
        user_mood = user_preferences.get('mood', all_moods[0])
        vector_part = [1 if user_mood.lower() == mood.lower() else 0 for mood in all_moods]
        user_vector += vector_part
    
    # Add genre one-hot encoding
    if all_genres:
        user_genres = user_preferences.get('genres', [])
        vector_part = [1 if genre.lower() in [g.lower() for g in user_genres] else 0 for genre in all_genres]
        user_vector += vector_part
    
    # Add numerical features
    mood_intensity = user_preferences.get('mood_intensity', 5) / 10  # Normalize to 0-1 scale
    user_vector.append(mood_intensity)
    
    stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
    stress_level = user_preferences.get('stress_level', 'Medium')
    if isinstance(stress_level, str):
        user_vector.append(stress_map.get(stress_level, 0.6))
    else:
        user_vector.append(stress_level)
    
    # Check if user vector dimensions match feature vectors
    if len(user_vector) == 0:
        print("WARNING: User vector is empty. Cannot compute recommendations.")
        return [{"title": "No recommendations available", "score": 0, "info": {}}]
    
    # Make sure dimensions match
    if len(user_vector) != feature_vectors.shape[1]:
        print(f"WARNING: User vector length ({len(user_vector)}) does not match feature vector length ({feature_vectors.shape[1]}). Adjusting...")
        # Pad with zeros or truncate
        if len(user_vector) < feature_vectors.shape[1]:
            user_vector += [0] * (feature_vectors.shape[1] - len(user_vector))
        else:
            user_vector = user_vector[:feature_vectors.shape[1]]
    
    # Calculate similarity
    print(f"User vector shape: {len(user_vector)}")
    print(f"Feature vectors shape: {feature_vectors.shape}")
    
    user_vector = np.array(user_vector).reshape(1, -1)
    similarity = cosine_similarity(user_vector, feature_vectors)[0]
    
    # Get top matches
    top_indices = similarity.argsort()[-top_n:][::-1]
    
    # Create recommendations list
    recommendations = []
    for idx in top_indices:
        movie = df.iloc[idx].to_dict()
        recommendations.append({
            'title': movie.get('Title', f"Movie {idx}"),
            'score': float(similarity[idx] * 100),
            'info': movie
        })
    
    return recommendations