import pandas as pd
import streamlit as st
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import json
# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to generate embeddings for movie overviews and create a FAISS vector database
def create_vector_database(metadata, embeddings_model=OpenAIEmbeddings()):
    texts = metadata['title'].tolist()
    vec_db = FAISS.from_texts(texts, embeddings_model)
    return vec_db

# Function to get recommendations based on user's watch history
def get_cf_ann_recommendations(watched_movies, vec_db, k=1000):

    # Get similar movies from their reviews embeddings
    relevant_movies_with_scores = vec_db.similarity_search_with_score(watched_movies, k=k)
    
    # Retrieve movie titles and scores
    recommended_movies = []
    for movie, score in relevant_movies_with_scores:
        recommended_movies.append({'title': movie.page_content, 
                                   'score': score}) 
    # Exclude wathced movies
    recommended_movies = [movie for movie in recommended_movies if movie['title'] not in watched_movies]
            
    recommendations_df = pd.DataFrame(recommended_movies)

    # Rank recommendations by score descendingly
    recommendations_df = recommendations_df.sort_values(by='score', ascending=False)
    
    return recommendations_df

def get_llm_recommendations(watched_movies, all_movies, model_name="gpt-3.5-turbo-1106"): #gpt-4-1106-preview, gpt-3.5-turbo-1106
    input_json = {
        'user_watched_movies': watched_movies,
        'all_movies': all_movies
    }
    print("Input to LLM: ", input_json)
    system_msg = """
       #Task:
        Act as a movie recommender system.I will give you a json like that:
        # Input:
        {
            'user_watched_movies': list of strings representing all the movies the user has watched,
            'all_movies': list of strings representing all the movies available in the database from which we can recommend movies,
        }

        # Output:
        I want you to return to me a json like that: 
        {
                'recommended_movies':[
                    list of json objects, each object represents a recommended movie including the following fields:
                    {   
                        'title': string representing the title of the recommended movie,
                        'score': string representing the score of the recommended movie, from 0 to 1, ranked based on relevance to the user,
                        'justification': string representing the reason provided by the LLM why this movie matches the user. It will help debugging and improve the prompt. limit your justification of why you picked this movie to 2 sentences maximum,
                    },
                ]
            }
        # Example Input:
        {
            'user_watched_movies':['Movie X', 'Movie Y'],
            'all_movies':['Movie 1', ..., 'Movie N'],
        }

        # Example Output:
        I want you to return to me a json like that: 
        {
                'recommended_movies':[
                    {
                    'title': 'Movie Z',
                    'score': '0.8',
                    'justification': 'Reason provided by the LLM why this movie matches the user. It will help debugging and improve the prompt',
                    },
                    {
                    'title': 'Movie P',
                    'score': '0.6',
                    'justification': 'Reason provided by the LLM why this movie matches the user. It will help debugging and improve the prompt',

                    }
                ]
            }

        #RULES:
        - Be specific with your recommendations. 
        - If the user is already watched a movie, don't recommend it, or give it score = 0
        - Pick recommendations only from the given set of all available movies 'all_movies'.
        - Movies that are sequels to movies the user has watched should be recommended with a higher score than other movies.
        - Provide your answer in json format only with no extra unformatted text so that I can parse it in code. 
        - Do not enclose your answer in ```json quotes
        """

    response = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": json.dumps(input_json)}],
        model=model_name,
        temperature=0.1,
        top_p=0.1,
        #Compatible with gpt-4-1106-preview and gpt-3.5-turbo-1106.
        response_format={"type": "json_object"},
    )

    recommendations = json.loads(response.choices[0].message.content)['recommended_movies']
    print("Recommendations from LLM: ", recommendations)
    # Exclude wathced movies
    recommendations = [movie for movie in recommendations if movie['title'] not in watched_movies]
    
    # Exclude movies with score < 0.5   
    recommendations = [rec for rec in recommendations if float(rec['score']) > 0.5]   
            
    recommendations_df = pd.DataFrame(recommendations)

    # Rank recommendations by score descendingly
    recommendations_df = recommendations_df.sort_values(by='score', ascending=False)
    
    return recommendations_df

def main():
    st.title("LLM-based Movie Recommender System")

    n_movies = 10000

    # Initialize embeddings model and create a FAISS vector database
    if 'vec_db' not in st.session_state:
        with st.spinner("Loading movie metadata..."):
            # Load Movies Metadata
            metadata = pd.read_csv('.\imdb.data\movies_metadata.csv')
            # Limit for memory purposes
            metadata = metadata[:n_movies]
            st.session_state.metadata = metadata
            

        with st.spinner("Creating vector database for movie embeddings..."):
            embeddings_model = OpenAIEmbeddings()
            st.session_state.vec_db = create_vector_database(st.session_state.metadata, embeddings_model)    
                
    
    # User input for watched movies
    user_history_input = st.text_area("Enter watched movies as a JSON list:", '["The Dark Knight", "Inception"]')
    watched_movies = json.loads(user_history_input)
    
    all_movies = get_cf_ann_recommendations(" ".join(watched_movies), st.session_state.vec_db, k=1000)['title'].tolist()

    # Get recommendations using LLM
    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations from LLM..."):
            recommendations = get_llm_recommendations(watched_movies, all_movies)

            st.write("LLM Recommended Movies:")
            st.write(recommendations)
            
if __name__ == "__main__":
    main()
