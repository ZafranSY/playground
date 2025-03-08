import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_feature_vectors(df):
    """Create feature vectors for movie attributes."""
    all_emotions = df['Emotion'].unique().tolist()
    all_moods = df['Mood'].unique().tolist()
    
    all_genres = []
    for genres in df['Genre']:
        all_genres.extend(genres)
    all_genres = list(set(all_genres))
    
    feature_vectors = []
    for _, row in df.iterrows():
        vector = []
        vector += [1 if row['Emotion'] == emotion else 0 for emotion in all_emotions]
        vector += [1 if row['Mood'] == mood else 0 for mood in all_moods]
        vector += [1 if genre in row['Genre'] else 0 for genre in all_genres]
        vector.append(row['Mood Intensity'] / 10 if 'Mood Intensity' in row else 0.5)
        stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
        vector.append(stress_map.get(row.get('Stress Level', 'Medium'), 0.6))
        feature_vectors.append(vector)
    
    return np.array(feature_vectors), all_emotions, all_moods, all_genres

def get_recommendations(df, user_input, feature_vectors, all_emotions, all_moods, all_genres, top_n=3):
    """Find movie recommendations based on user preferences."""
    user_vector = []
    user_vector += [1 if user_input['emotion'] == emotion else 0 for emotion in all_emotions]
    user_vector += [1 if user_input['mood'] == mood else 0 for mood in all_moods]
    user_vector += [1 if genre in user_input.get('genres', []) else 0 for genre in all_genres]
    user_vector.append(user_input.get('mood_intensity', 5) / 10)
    stress_map = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}
    user_vector.append(stress_map.get(user_input.get('stress_level', 'Medium'), 0.6))
    user_vector = np.array(user_vector).reshape(1, -1)
    
    similarity = cosine_similarity(user_vector, feature_vectors)[0]
    top_indices = similarity.argsort()[-top_n:][::-1]
    
    recommendations = []
    for idx in top_indices:
        movie = df.iloc[idx].to_dict()
        recommendations.append({
            'title': movie['Title'],
            'score': float(similarity[idx] * 100),
            'info': movie
        })
    
    return recommendations
