import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten

# Sample Dataset
import json

data = [
    {"Title": "Ejen Ali: The Movie 2", "Year": 2025, "Duration": "2h", "Genre": ["Animation"], "Emotion": "Exciting", "Mood": "Thrilling", "Suggested Watch Time": "Weekend Afternoon"},
    {"Title": "Don't Look at the Demon", "Year": 2022, "Duration": "1h 35m", "Genre": ["Horror"], "Emotion": "Scary", "Mood": "Dark", "Suggested Watch Time": "Late Night"},
    {"Title": "Tiger Stripes", "Year": 2023, "Duration": "1h 35m", "Genre": ["Drama"], "Emotion": "Coming-of-Age", "Mood": "Unsettling", "Suggested Watch Time": "Quiet Evening"},
    # Add more entries from the dataset
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing
mlb_genre = MultiLabelBinarizer()
df_genre = pd.DataFrame(mlb_genre.fit_transform(df['Genre']), columns=mlb_genre.classes_)
df = pd.concat([df, df_genre], axis=1)

# Encode categorical fields
emotion_mapping = {label: idx for idx, label in enumerate(df['Emotion'].unique())}
mood_mapping = {label: idx for idx, label in enumerate(df['Mood'].unique())}

# Map emotions and moods to numeric values
df['Emotion'] = df['Emotion'].map(emotion_mapping)
df['Mood'] = df['Mood'].map(mood_mapping)

# Feature and Target Split
X = df[df_genre.columns.tolist() + ['Emotion', 'Mood']]
y = pd.get_dummies(df['Title'])  # Target is the title (one-hot encoded)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build TensorFlow Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save Model
model.save('recommender_model.h5')

# Prediction Function
def recommend(content_features):
    # Example input: {'Genre': ['Animation'], 'Emotion': 'Exciting', 'Mood': 'Thrilling'}
    user_input = np.zeros((1, X.shape[1]))
    user_input[0, :-2] = mlb_genre.transform([content_features['Genre']])[0]
    user_input[0, -2] = emotion_mapping[content_features['Emotion']]
    user_input[0, -1] = mood_mapping[content_features['Mood']]
    
    predictions = model.predict(user_input)
    recommended_index = np.argmax(predictions)
    return y.columns[recommended_index]

# Example Usage
example_input = {
    "Genre": ["Animation"],
    "Emotion": "Exciting",
    "Mood": "Thrilling"
}
print("Recommended Title:", recommend(example_input))
