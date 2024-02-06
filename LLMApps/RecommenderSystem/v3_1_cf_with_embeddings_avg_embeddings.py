import pandas as pd
import numpy as np
import streamlit as st
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

# Function to compute the weighted rating of each movie
def weighted_rating(x, M, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+M) * R) + (M/(M+v) * C)

# Function to get the top-5 movie recommendations
def get_simple_recommendations(metadata):
    C = metadata['vote_average'].mean()
    M = metadata['vote_count'].quantile(0.90)

    q_movies = metadata.copy().loc[metadata['vote_count'] >= M]
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, args=(M, C))
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies[['title', 'vote_count', 'vote_average', 'score']].head(5)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", "Empty")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Function to generate embeddings for movie overviews
def calculate_movies_features(metadata):
    metadata['overview'] = metadata['overview'].fillna('Invalid')
    #embeddings = [get_embedding(overview) for overview in metadata['overview']]
    embeddings = []
    for overview in metadata['overview'].tolist():

        try:
            embeddings.append(get_embedding(overview))
            
        except Exception as e:
            print("Error: ", e, " in movie overview: ", overview)
            continue  
    return np.array(embeddings)

# Function to calculate user profile embeddings
def calculate_user_features(watched_movies, metadata, emb_len):
    watched_embeddings = []
    for movie in watched_movies:
        if movie in metadata['title'].values:
            overview = metadata.loc[metadata['title'] == movie, 'overview'].iloc[0]
            watched_embeddings.append(get_embedding(overview))
    if len(watched_embeddings) == 0:        
        #user_profile = np.zeros(emb_len)
        return None, False
    else:
        user_profile = np.mean(watched_embeddings, axis=0)
    return user_profile, True

# Function to get recommendations using cosine similarity
def get_recommendations(watched_movies, movie_embeddings, metadata):
    emb_len = movie_embeddings.shape[1]
    user_profile, status = calculate_user_features(watched_movies, metadata, emb_len)
    if status == False:
        st.write("Cannot find the movie titles entered, falling back to default recommendations")
        return get_simple_recommendations(metadata)
    
    cosine_sim = cosine_similarity(np.array([user_profile]), movie_embeddings)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude movies already watched
    movie_indices = [i[0] for i in sim_scores if metadata['title'].iloc[i[0]] not in watched_movies]

    # Get top 5 recommendations
    top_recommendations = metadata['title'].iloc[movie_indices][:5]

    # The sim_scores are tuples of (index, score), so we process score[1] for the score
    recommendations = [{'title': title, 'score': score[1]} for title, score in zip(top_recommendations, sim_scores)]
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df



def main():
    st.title("Collaborative Filtering Movie Recommender System with Embeddings")


    n_movies = 1000
    # Calculate movie embeddings
    if 'movie_embeddings' not in st.session_state:
        with st.spinner("Loading movie metadata..."):
            start_time = time.time()
            # Load Movies Metadata
            metadata = pd.read_csv('.\imdb.data\movies_metadata.csv')

            # Limit for memory purposes
            metadata = metadata[:n_movies]
            st.session_state.metadata = metadata
            end_time = time.time()
            time_spent = end_time-start_time

            st.write(f"Time to load the data is {time_spent} seconds")
        with st.spinner("Calculating movie embeddings..."):
            start_time = time.time()
            st.session_state.movie_embeddings = calculate_movies_features(st.session_state.metadata)
            end_time = time.time()
            time_spent = end_time-start_time
            st.write(f"Time to calculate embeddings {time_spent} seconds for {n_movies} movies")

    st.write("Sample metadata:")
    st.write(st.session_state.metadata[['title', 'overview']].head(10))

    # User input for watched movies
    user_history_input = st.text_area("Enter watched movies as a JSON list:", '["Toy Story", "Jumanji"]')
    watched_movies = json.loads(user_history_input)

    # Calculate user profile and get recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Calculating recommendations..."):
            start_time = time.time()
            recommendations = get_recommendations(watched_movies, st.session_state.movie_embeddings, st.session_state.metadata)
            end_time = time.time()
            time_spent = end_time-start_time
            st.write(f"Time to get recommendations is {time_spent} seconds")
            st.write("Recommended Movies:")
            st.write(recommendations)

if __name__ == "__main__":
    main()
