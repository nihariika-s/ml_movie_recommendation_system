import pandas as pd
import pickle

print("=== TASK 3: Fast Model Building (No Warnings) ===")

ratings_path = "data/ratings_clean.csv"
ratings = pd.read_csv(ratings_path)

# -----------------------------------------
# 1. FAST user → movie dictionary
# -----------------------------------------
print("Building user → movie dictionary...")

user_ratings = (
    ratings.groupby("userId")[["movieId", "rating"]]
    .apply(lambda x: dict(zip(x["movieId"], x["rating"])))
    .to_dict()
)

# -----------------------------------------
# 2. FAST movie → user dictionary
# -----------------------------------------
print("Building movie → user dictionary...")

movie_ratings = (
    ratings.groupby("movieId")[["userId", "rating"]]
    .apply(lambda x: dict(zip(x["userId"], x["rating"])))
    .to_dict()
)

# -----------------------------------------
# 3. Save results
# -----------------------------------------
print("Saving model files...")

pickle.dump(user_ratings, open("data/user_ratings.pkl", "wb"))
pickle.dump(movie_ratings, open("data/movie_ratings.pkl", "wb"))

print("=== Fast Model Saved Successfully ===")