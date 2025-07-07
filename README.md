# Music-Recommendation-Generator
This project develops a personalized music recommendation system using collaborative and content-based filtering. By analyzing user behavior and song features, it suggests relevant tracks, enhancing user experience through real-world applications of machine learning and data analysis.

# 🎧 Music Recommendation System

This project is a Python-based music recommendation system that suggests songs to users based on their preferences and listening history using **Collaborative Filtering** and **Content-Based Filtering**.

---

## 📌 Features

- 🎵 Recommend songs based on user-song ratings.
- 🤝 Collaborative filtering using matrix factorization (SVD).
- 🧠 Content-based filtering using TF-IDF on song metadata.
- 📊 Sample data included for easy testing and demonstration.
- 🧪 Simple and modular structure for expansion and experiments.

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- Surprise (for collaborative filtering)
- TF-IDF & Cosine Similarity

---

## 📂 Project Structure

music-recommendation/
├── main.py # Entry point
├── requirements.txt # Project dependencies
├── data/
│ ├── users.csv # User list
│ ├── songs.csv # Song metadata
│ └── interactions.csv # User-song ratings
└── src/
├── data_preprocessing.py # Data loading and preprocessing
├── collaborative.py # Collaborative filtering logic
└── content_based.py # Content-based filtering logic

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ananya_Arya/music-recommendation.git
   cd music-recommendation
   pip install -r requirements.txt
   python main.py
   🎵 Music Recommendation System 🎵

Top 5 collaborative recommendations for User 1:
- Song ID: 1 (Score: 4.76)
- Song ID: 3 (Score: 4.59)
...

Top 5 songs similar to 'Shape of You':
- Happier
- Blinding Lights
- Perfect
...
🎵 Music Recommendation System 🎵

Top 5 collaborative recommendations for User 1:
- Song ID: 1 (Score: 4.76)
- Song ID: 3 (Score: 4.59)
...

Top 5 songs similar to 'Shape of You':
- Happier
- Blinding Lights
- Perfect
...
🧪 Future Improvements
Integrate Spotify/Youtube API for real song playback.

Use deep learning (autoencoders, embeddings).

Add a Flask/Streamlit web interface.

Real-time feedback loop for dynamic recommendations.

📜 License
This project is open-source and available under the MIT License.

---

Let me know if you'd like a web UI version (using Flask or Streamlit) too!

  

