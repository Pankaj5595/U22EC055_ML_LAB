# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess ratings
rating_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
rating_stats.columns = ['movieId', 'avg_rating', 'rating_count']
movies = pd.merge(movies, rating_stats, on='movieId', how='left')
movies['genres'] = movies['genres'].fillna('')
movies['avg_rating'] = movies['avg_rating'].fillna(0)
movies['rating_count'] = movies['rating_count'].fillna(0)
movies['processed_genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Genre vectorization
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['processed_genres'])
cosine_sim = cosine_similarity(genre_matrix)

# Reverse index
title_to_index = pd.Series(movies.index, index=movies['title'])

# Recommendation logic
def recommend_movies(title, top_n=10):
    if title not in title_to_index:
        return None
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices][['title', 'avg_rating', 'rating_count']]
    similar_movies = similar_movies.sort_values(by=['avg_rating', 'rating_count'], ascending=[False, False])
    return similar_movies.reset_index(drop=True)

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie name and get similar, highly-rated movie recommendations.")

movie_list = sorted(movies['title'].dropna().unique())
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    result = recommend_movies(selected_movie)
    if result is not None:
        st.subheader("Recommended Movies:")
        st.dataframe(result)
    else:
        st.warning("Movie not found in the database.")
