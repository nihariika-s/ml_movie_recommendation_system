import pandas as pd

print("=== TASK 1: Loading Data ===")

ratings_path = "data/ratings.csv"
movies_path  = "data/movies.csv"
tags_path    = "data/tags.csv"

# Load files
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)

print("Files loaded successfully!")
print(f"Movies shape: {movies.shape}")
print(f"Ratings shape: {ratings.shape}")
print(f"Tags shape: {tags.shape}")

# Save cleaned versions
ratings.to_csv("data/ratings_clean.csv", index=False)
movies.to_csv("data/movies_clean.csv", index=False)
tags.to_csv("data/tags_clean.csv", index=False)

print("Cleaned files saved successfully!")