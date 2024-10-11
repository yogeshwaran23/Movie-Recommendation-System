# Importing the required libraries
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Loading the dataset
movies_data = pd.read_csv('movies.csv')

# Checking the first few rows
print(movies_data.head())

# Selecting relevant features for content-based filtering
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Filling missing values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining the selected features into one column
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorizing the combined features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
feature_vectors = tfidf_vectorizer.fit_transform(combined_features)

# Calculating the cosine similarity between movies
cosine_sim = cosine_similarity(feature_vectors)

# Collaborative Filtering using Matrix Factorization (SVD)
# Create a user-item matrix (example using a ratings dataset)
ratings_data = pd.read_csv('ratings.csv')
user_movie_matrix = ratings_data.pivot_table(index='userId', columns='movieId', values='rating')

# Filling NaN values with 0 (can be adjusted based on requirements)
user_movie_matrix.fillna(0, inplace=True)

# Applying SVD (Matrix Factorization) for collaborative filtering
svd = TruncatedSVD(n_components=20)
matrix_decomposed = svd.fit_transform(user_movie_matrix)

# Now, for each user, we can predict ratings for unwatched movies
predicted_ratings = np.dot(matrix_decomposed, svd.components_)

# Function to recommend movies based on a given movie
def recommend_movie_based_on_title(movie_title, similarity_matrix, movie_data, top_n=10):
    # Find the closest match to the movie title
    movie_titles = movie_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_title, movie_titles, n=1)
    
    if not close_match:
        return "Movie not found in the database."
    
    # Get the index of the matched movie
    movie_index = movie_data[movie_data.title == close_match[0]]['index'].values[0]
    
    # Get similarity scores for the movie
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort the movies based on similarity scores
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Print top N recommended movies
    print(f"Movies similar to '{movie_title}':")
    recommendations = []
    for i in range(1, top_n + 1):
        movie_idx = sorted_movies[i][0]
        recommendations.append(movie_data['title'].iloc[movie_idx])
    
    return recommendations

# Hybrid Recommendation System: combine collaborative filtering and content-based
def hybrid_recommendation(user_id, movie_title, similarity_matrix, movie_data, user_movie_pred, top_n=10):
    # Content-based filtering recommendation (based on movie title)
    content_based_recommendations = recommend_movie_based_on_title(movie_title, similarity_matrix, movie_data, top_n)
    
    # Collaborative filtering recommendations (based on user preferences)
    user_pred_ratings = user_movie_pred[user_id]
    user_recommendations = np.argsort(user_pred_ratings)[::-1]  # Sort movies by predicted ratings
    
    print(f"Hybrid recommendations for user {user_id} based on '{movie_title}':")
    hybrid_recommendations = []
    
    for idx in user_recommendations[:top_n]:
        movie_name = movie_data[movie_data['movieId'] == idx]['title'].values[0]
        hybrid_recommendations.append(movie_name)
    
    return hybrid_recommendations

# Example of using the system
movie_name = "The Dark Knight"
user_id = 5
top_n = 10

# Generating content-based recommendations
content_recommendations = recommend_movie_based_on_title(movie_name, cosine_sim, movies_data, top_n)
print("Content-based Recommendations:", content_recommendations)

# Generating hybrid recommendations (content + collaborative)
hybrid_recommendations = hybrid_recommendation(user_id, movie_name, cosine_sim, movies_data, predicted_ratings, top_n)
print("Hybrid Recommendations:", hybrid_recommendations)
