# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import pandas as pd
# # from recommendation_model.data1 import load_movie_data
# # from model import create_feature_vectors, get_recommendations

# # app = FastAPI()

# # # Load and preprocess movie data
# # movie_data = load_movie_data()
# # df = pd.DataFrame(movie_data)
# # feature_vectors, all_emotions, all_moods, all_genres = create_feature_vectors(df)

# # class UserInput(BaseModel):
# #     emotion: str
# #     mood: str
# #     mood_intensity: int
# #     stress_level: str
# #     genres: list[str]

# # @app.get("/")
# # def root():
# #     return {"message": "Welcome to the Movie Mood Matcher API"}

# # @app.get("/options")
# # def get_options():
# #     """Return available emotions, moods, and genres."""
# #     return {
# #         "emotions": all_emotions,
# #         "moods": all_moods,
# #         "genres": all_genres
# #     }

# # @app.post("/recommend")  # Change from "/recommendations" to "/recommend"
# # def recommendations(user_input: UserInput):
# #     """Get movie recommendations based on user input."""
# #     recs = get_recommendations(
# #         df, 
# #         user_input.dict(), 
# #         feature_vectors, 
# #         all_emotions, 
# #         all_moods, 
# #         all_genres
# #     )
# #     return {"recommendations": recs}
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import NearestNeighbors
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate

# def create_feature_vectors(df):
#     """
#     Create feature vectors using the movies DataFrame
#     Returns the feature vectors and lists of emotions, moods, and genres
#     """
#     # Extract unique emotions, moods, and genres
#     all_emotions = df['Emotion'].unique().tolist() if 'Emotion' in df.columns else []
#     all_moods = df['Mood'].unique().tolist() if 'Mood' in df.columns else []
    
#     # Extract all unique genres from the list of genres for each movie
#     all_genres = []
#     if 'Genre' in df.columns:
#         for genres in df['Genre']:
#             if isinstance(genres, list):
#                 all_genres.extend(genres)
#         all_genres = list(set(all_genres))
    
#     # Create feature vectors for each movie
#     feature_vectors = []
#     for _, row in df.iterrows():
#         vector = []
        
#         # Add emotion one-hot encoding
#         if 'Emotion' in df.columns:
#             vector += [1 if row['Emotion'] == emotion else 0 for emotion in all_emotions]
        
#         # Add mood one-hot encoding
#         if 'Mood' in df.columns:
#             vector += [1 if row['Mood'] == mood else 0 for mood in all_moods]
        
#         # Add genre one-hot encoding
#         if 'Genre' in df.columns:
#             vector += [1 if genre in row['Genre'] else 0 for genre in all_genres]
        
#         # Add numerical features
#         if 'Mood Intensity' in df.columns:
#             vector.append(row['Mood Intensity'] / 10)  # Normalize to 0-1 scale
        
#         if 'Stress Level' in df.columns:
#             stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
#             if isinstance(row.get('Stress Level'), str):
#                 vector.append(stress_map.get(row.get('Stress Level', 'Medium'), 0.6))
#             else:
#                 vector.append(row.get('Stress Level', 0.6))
        
#         feature_vectors.append(vector)
    
#     return np.array(feature_vectors), all_emotions, all_moods, all_genres

# def build_content_based_model(movies_features):
#     """
#     Build a content-based recommendation model using cosine similarity
#     """
#     # Extract feature matrix (excluding movie_id and Title columns)
#     feature_matrix = movies_features.drop(['movie_id', 'Title'], axis=1, errors='ignore').values
    
#     # Compute similarity matrix
#     similarity_matrix = cosine_similarity(feature_matrix)
    
#     # Create a model using K-nearest neighbors
#     model = NearestNeighbors(metric='cosine', algorithm='brute')
#     model.fit(feature_matrix)
    
#     return model, similarity_matrix, feature_matrix

# def build_collaborative_model(data, num_users, num_movies, embedding_size=50):
#     """
#     Build a neural network-based collaborative filtering model
#     """
#     # User input
#     user_input = Input(shape=(1,), name='user_input')
#     user_embedding = Embedding(num_users + 1, embedding_size, name='user_embedding')(user_input)
#     user_vec = Flatten(name='flatten_users')(user_embedding)
    
#     # Movie input
#     movie_input = Input(shape=(1,), name='movie_input')
#     movie_embedding = Embedding(num_movies + 1, embedding_size, name='movie_embedding')(movie_input)
#     movie_vec = Flatten(name='flatten_movies')(movie_embedding)
    
#     # Feature input (for movie features)
#     feature_cols = len(data.columns) - 3 if len(data.columns) > 3 else 1  # Excluding user_id, movie_id, rating
#     feature_input = Input(shape=(feature_cols,), name='feature_input')
#     feature_dense = Dense(100, activation='relu')(feature_input)
    
#     # Concatenate layers
#     concat = Concatenate()([user_vec, movie_vec, feature_dense])
    
#     # Dense layers
#     dense1 = Dense(128, activation='relu')(concat)
#     dense2 = Dense(64, activation='relu')(dense1)
#     output = Dense(1)(dense2)
    
#     # Create model
#     model = Model(
#         inputs=[user_input, movie_input, feature_input],
#         outputs=output
#     )
#     model.compile(
#         optimizer='adam',
#         loss='mean_squared_error'
#     )
    
#     return model

# class HybridRecommender:
#     def __init__(self, content_model, collaborative_model, movies_features, similarity_matrix, feature_matrix, all_emotions, all_moods, all_genres):
#         self.content_model = content_model
#         self.collaborative_model = collaborative_model
#         self.movies_features = movies_features
#         self.similarity_matrix = similarity_matrix
#         self.feature_matrix = feature_matrix
#         self.all_emotions = all_emotions
#         self.all_moods = all_moods
#         self.all_genres = all_genres
        
#     def get_content_recommendations(self, movie_idx, n=10):
#         """Get content-based recommendations"""
#         # Get similarity scores
#         movie_similarities = self.similarity_matrix[movie_idx]
        
#         # Get top similar movies
#         similar_movies_indices = movie_similarities.argsort()[-n-1:-1][::-1]
#         similar_movies = self.movies_features.iloc[similar_movies_indices]
        
#         return similar_movies
    
#     def get_user_recommendations(self, user_id, user_features, n=10):
#         """
#         Get personalized recommendations for a user
#         combining collaborative and content-based filtering
#         """
#         # Create user feature vector
#         user_vector = []
        
#         # Add emotion one-hot encoding
#         user_vector += [1 if user_features['emotion'] == emotion else 0 for emotion in self.all_emotions]
        
#         # Add mood one-hot encoding
#         user_vector += [1 if user_features['mood'] == mood else 0 for mood in self.all_moods]
        
#         # Add genre one-hot encoding
#         user_vector += [1 if genre in user_features.get('genres', []) else 0 for genre in self.all_genres]
        
#         # Add numerical features
#         user_vector.append(user_features.get('mood_intensity', 5) / 10)  # Normalize to 0-1 scale
        
#         stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
#         user_vector.append(stress_map.get(user_features.get('stress_level', 'Medium'), 0.6))
        
#         # Get content-based recommendations
#         user_vector_np = np.array(user_vector).reshape(1, -1)
#         distances, indices = self.content_model.kneighbors(user_vector_np, n_neighbors=min(n, len(self.movies_features)))
#         content_recommendations = self.movies_features.iloc[indices[0]]
        
#         # Get collaborative recommendations if user exists in training data
#         # (simplified here - in practice would use the collaborative model)
        
#         # Combine recommendations (simplified approach)
#         final_recommendations = content_recommendations
        
#         return final_recommendations

# def get_recommendations(df, user_preferences, feature_vectors, all_emotions, all_moods, all_genres, top_n=5):
#     """Find movie recommendations based on user preferences."""
#     user_vector = []
    
#     # Add emotion one-hot encoding
#     user_vector += [1 if user_preferences['emotion'] == emotion else 0 for emotion in all_emotions]
    
#     # Add mood one-hot encoding
#     user_vector += [1 if user_preferences['mood'] == mood else 0 for mood in all_moods]
    
#     # Add genre one-hot encoding
#     user_vector += [1 if genre in user_preferences.get('genres', []) else 0 for genre in all_genres]
    
#     # Add numerical features
#     user_vector.append(user_preferences.get('mood_intensity', 5) / 10)  # Normalize to 0-1 scale
    
#     stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
#     user_vector.append(stress_map.get(user_preferences.get('stress_level', 'Medium'), 0.6))
    
#     # Calculate similarity
#     user_vector = np.array(user_vector).reshape(1, -1)
#     similarity = cosine_similarity(user_vector, feature_vectors)[0]
    
#     # Get top matches
#     top_indices = similarity.argsort()[-top_n:][::-1]
    
#     # Create recommendations list
#     recommendations = []
#     for idx in top_indices:
#         movie = df.iloc[idx].to_dict()
#         recommendations.append({
#             'title': movie['Title'],
#             'score': float(similarity[idx] * 100),
#             'info': movie
#         })
    
#     return recommendations
#thi sis main.py
import argparse
from data import load_movie_data, prepare_data
from model import create_feature_vectors, build_content_based_model, build_collaborative_model, HybridRecommender

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--data", help="Path to movie data JSON file", default=None)
    parser.add_argument("--ratings", help="Path to user ratings CSV file", default=None)
    args = parser.parse_args()
    
    # Load movie data
    movie_data = load_movie_data() if not args.data else load_movie_data(args.data)
    
    # Prepare data
    data, movies_features, feature_columns = prepare_data(movie_data, args.ratings)
    
    # Create feature vectors
    feature_vectors, all_emotions, all_moods, all_genres = create_feature_vectors(movies_features)
    
    # Build content-based model
    content_model, similarity_matrix, feature_matrix = build_content_based_model(movies_features)
    
    # Build collaborative model if we have user ratings
    collaborative_model = None
    if not data.empty and len(data) > 1:
        num_users = data['user_id'].max()
        num_movies = data['movie_id'].max()
        collaborative_model = build_collaborative_model(data, num_users, num_movies)
    
    # Create recommender
    recommender = HybridRecommender(
        content_model,
        collaborative_model,
        movies_features,
        similarity_matrix,
        feature_matrix,
        all_emotions,
        all_moods,
        all_genres
    )
    
    # Example user preferences
    user_preferences = {
        'emotion': 'Intense',
        'mood': 'Gripping',
        'mood_intensity': 8,  # Scale 1-10
        'stress_level': 'Medium',
        'genres': ['Drama', 'Crime']
    }
    
    # Get recommendations
    recommendations = recommender.get_user_recommendations(
        user_id=1,  # Example user ID
        user_features=user_preferences,
        n=5
    )
    
    print("\nRecommended movies based on preferences:")
    for i, movie in recommendations.iterrows():
        print(f"- {movie['Title']}")
    
    # Interactive mode
    while True:
        print("\nEnter your preferences (or type 'exit' to quit):")
        emotion = input("Emotion (e.g., Happy, Sad, Intense): ")
        if emotion.lower() == 'exit':
            break
            
        mood = input("Mood (e.g., Uplifting, Depressing, Gripping): ")
        mood_intensity = int(input("Mood Intensity (1-10): "))
        stress_level = input("Stress Level (Low, Medium, High): ")
        genres = input("Genres (comma-separated, e.g., Drama,Comedy): ").split(',')
        
        user_preferences = {
            'emotion': emotion,
            'mood': mood,
            'mood_intensity': mood_intensity,
            'stress_level': stress_level,
            'genres': [g.strip() for g in genres]
        }
        
        # Get recommendations
        recommendations = recommender.get_user_recommendations(
            user_id=1,
            user_features=user_preferences,
            n=5
        )
        
        print("\nRecommended movies based on your preferences:")
        for i, movie in recommendations.iterrows():
            print(f"- {movie['Title']}")

if __name__ == "__main__":
    main()