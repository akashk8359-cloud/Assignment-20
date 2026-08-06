import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Handle missing values
df["overview"] = df["overview"].fillna("")

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["clean_text"] = df["overview"].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

# Cosine Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommendation Function
def recommend(movie_name, top_n=5):
    index = df[df["title"] == movie_name].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i]["title"] for i, score in scores]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Select a Movie", df["title"].values)

if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    for m in recommend(movie):
        st.write("✅", m)