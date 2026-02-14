import pandas as pd

print("=== TASK 2: Feature Engineering ===")

ratings = pd.read_csv("data/ratings_clean.csv")
movies = pd.read_csv("data/movies_clean.csv")

# Extract genres as separate columns
movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))

# Movie popularity (number of ratings per movie)
movie_popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')

# Merge popularity with movies
movies = movies.merge(movie_popularity, on='movieId', how='left')

# Save engineered dataset
movies.to_csv("data/movies_engineered.csv", index=False)

print("Feature engineering completed & saved!")