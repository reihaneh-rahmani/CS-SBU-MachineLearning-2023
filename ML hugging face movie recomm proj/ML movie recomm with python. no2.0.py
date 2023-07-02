"""ML Movie Recommendation with Python. no2.0"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import pipeline, set_seed
import gradio as gr
import requests


# Load the Movies Dataset
movies_df = pd.read_csv('movies.csv')

# Select relevant columns from the dataset
movies = movies_df[['title', 'overview', 'genres', 'release_date', 'vote_average']]

# Clean the data
movies.dropna(inplace=True)

# Define a function to extract the genres of each movie
def get_genres(genres):
    genres_list = []
    for genre in eval(genres):
        genres_list.append(genre['name'])
    return genres_list

# Apply the get_genres function to the genres column
movies['genres'] = movies['genres'].apply(get_genres)


# Define a function to preprocess the data for clustering
def preprocess_data(movies):
    # Vectorize the movie overviews using the Text Generation pipeline from Hugging Face Transformers
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    overviews = movies['overview'].to_list()
    vectors = generator(overviews, max_length=50, num_return_sequences=1, do_sample=False)
    embeddings = [vector['generated_text'] for vector in vectors]

    # One-hot encode the movie genres
    genres = movies['genres'].explode().unique()
    for genre in genres:
        movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

    # Normalize the release dates and vote averages
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['release_date'] = movies['release_date'].apply(lambda x: x.year)
    movies['vote_average'] = (movies['vote_average'] - movies['vote_average'].mean()) / movies['vote_average'].std()

    # Combine the embeddings and one-hot encoded genres into a single feature matrix
    features = np.concatenate((np.array(embeddings), movies[genres]), axis=1)

    return features


# Preprocess the data
features = preprocess_data(movies)

# Cluster the data using KMeans
kmeans = KMeans(n_clusters=10, random_state=42).fit(features)
movies['cluster'] = kmeans.labels_


# Define a function to generate movie recommendations based on user input
def get_recommendations(movie_titles):
    # Get the cluster labels of the input movies
    input_movies = movies[movies['title'].isin(movie_titles)]
    input_clusters = input_movies['cluster'].unique()

    # Get the movies from the same clusters as the input movies
    cluster_movies = movies[movies['cluster'].isin(input_clusters)]
    cluster_movies = cluster_movies[~cluster_movies['title'].isin(movie_titles)]

    # Calculate the distance between the input movies and the movies in each cluster
    distances = pairwise_distances_argmin_min(kmeans.cluster_centers_[input_clusters], cluster_movies.iloc[:, 5:])

    # Get the indices of the closest movies in each cluster
    closest_indices = distances[0]

    # Get the recommended movies
    recommendations = cluster_movies.iloc[closest_indices][['title', 'overview']].to_dict('records')

    return recommendations


# Define a Gradio interface for the model
iface = gr.Interface(
    fn=get_recommendations,
    inputs=gr.inputs.MultiText(label="Enter at least three movie titles"),
    outputs=gr.outputs.Table(columns=["title", "overview"]),
    title="Movie Recommender System"
)


# Launch the Gradio interface
iface.launch()

# Define a Streamlit interface for the model
st.title("Movie Recommender System")

# Get user input
movie_titles = st.multiselect("Enter at least three movie titles", movies['title'].unique())

# Get recommendations
if len(movie_titles) >= 3:
    recommendations = get_recommendations(movie_titles)
    st.table(recommendations)
else:
    st.write("Please enter at least three movie titles")

# Deploy the model and interface on HuggingFace Spaces
model_url = "https://api-inference.huggingface.co/models/username/movie-recommender"
headers = {"Authorization": "Bearer api_key"}

def get_predictions(movie_titles):
    data = {"inputs": movie_titles}
    response = requests.post(model_url, headers=head
ers, json=data)
    return response.json()

app = gr.Interface(fn=get_predictions, inputs=gr.inputs.MultiText(label="Enter at least three movie titles"), outputs=gr.outputs.Table(columns=["title", "overview"]))

app.run()