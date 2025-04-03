import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("🎬 Movie Recommendation System")
st.write("Enter your favorite movie, and we'll suggest similar ones!")




movies_data = pd.read_csv('movies.csv')


required_columns = ['title', 'genres', 'keywords', 'tagline', 'cast', 'director']
if not all(col in movies_data.columns for col in required_columns):
    st.error("Dataset missing required columns!")
    st.stop()


for feature in required_columns[1:]:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)


movie_name = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    if movie_name:
        list_of_all_titles = movies_data['title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

        if close_match:
            matched_movie = close_match[0]
            index = movies_data[movies_data.title == matched_movie].index[0]
            similarity_scores = list(enumerate(similarity[index]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

            st.subheader("Recommended Movies:")
            for i, movie in sorted_movies:
                st.write(f"- {movies_data.iloc[i]['title']}")
        else:
            st.error("Movie not found in dataset. Try another!")
    else:
        st.warning("Please enter a movie name!")