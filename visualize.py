import pandas as pd
import matplotlib.pyplot as plt

print("=== TASK 4: Visualization ===")

movies = pd.read_csv("data/movies_engineered.csv")

# Top 20 movies by rating count
top_movies = movies.nlargest(20, "rating_count")

plt.figure(figsize=(12, 6))
plt.barh(top_movies["title"], top_movies["rating_count"])
plt.xlabel("Number of Ratings")
plt.title("Top 20 Most Rated Movies (MovieLens 32M)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Visualization complete!")