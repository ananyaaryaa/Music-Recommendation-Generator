### File: main.py
from src.collaborative import CollaborativeFiltering
from src.content_based import ContentBasedFiltering
from src.data_preprocessing import load_data, preprocess_data


def main():
    print("\n🎵 Music Recommendation System 🎵\n")

    # Load and preprocess data
    user_data, song_data, interactions = load_data()
    user_data, song_data, interactions = preprocess_data(user_data, song_data, interactions)

    # Initialize models
    collab_model = CollaborativeFiltering(interactions)
    content_model = ContentBasedFiltering(song_data)

    # Example recommendations
    user_id = 1
    print(f"Top 5 collaborative recommendations for User {user_id}:")
    print(collab_model.recommend(user_id))

    song_title = "Shape of You"
    print(f"\nTop 5 songs similar to '{song_title}':")
    print(content_model.recommend(song_title))


if __name__ == "__main__":
    main()


### File: src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    user_data = pd.read_csv("data/users.csv")
    song_data = pd.read_csv("data/songs.csv")
    interactions = pd.read_csv("data/interactions.csv")
    return user_data, song_data, interactions

def preprocess_data(user_data, song_data, interactions):
    le_user = LabelEncoder()
    le_song = LabelEncoder()
    interactions['user'] = le_user.fit_transform(interactions['user'])
    interactions['song'] = le_song.fit_transform(interactions['song'])
    return user_data, song_data, interactions


### File: src/collaborative.py
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class CollaborativeFiltering:
    def __init__(self, interactions_df):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(interactions_df[['user', 'song', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2)
        self.model = SVD()
        self.model.fit(trainset)
        self.song_ids = interactions_df['song'].unique()

    def recommend(self, user_id, top_n=5):
        predictions = [(song_id, self.model.predict(user_id, song_id).est) for song_id in self.song_ids]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [f"Song ID: {pred[0]} (Score: {pred[1]:.2f})" for pred in predictions[:top_n]]


### File: src/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedFiltering:
    def __init__(self, song_data):
        self.song_data = song_data
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(song_data['metadata'])
        self.indices = pd.Series(song_data.index, index=song_data['title']).drop_duplicates()

    def recommend(self, title, top_n=5):
        idx = self.indices.get(title)
        if idx is None:
            return ["Song not found."]
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        return [self.song_data.iloc[i[0]]['title'] for i in sim_scores]


### File: data/users.csv
user
Alice
Bob
Charlie


### File: data/songs.csv
title,metadata
Shape of You,"pop ed sheeran upbeat love"
Blinding Lights,"pop weeknd synth love fast"
Perfect,"ballad ed sheeran romantic slow"
Starboy,"rnb weeknd daftpunk dark"
Happier,"pop marshmello upbeat heartbreak"


### File: data/interactions.csv
user,song,rating
Alice,Shape of You,5
Alice,Perfect,4
Bob,Blinding Lights,5
Bob,Starboy,3
Charlie,Happier,4
Charlie,Shape of You,2


### File: requirements.txt
pandas
scikit-learn
surprise
numpy
