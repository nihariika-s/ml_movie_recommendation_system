import pickle
from collections import defaultdict

# Load files
user_ratings = pickle.load(open("data/user_ratings.pkl", "rb"))
movie_ratings = pickle.load(open("data/movie_ratings.pkl", "rb"))

# ---------------------------------------------------
# Movie similarity score (Fast)
# ---------------------------------------------------
def movie_similarity(movie1, movie2):
    users1 = movie_ratings.get(movie1, {})
    users2 = movie_ratings.get(movie2, {})

    common_users = set(users1.keys()) & set(users2.keys())

    if len(common_users) == 0:
        return 0

    # dot-product based similarity
    total = sum(users1[u] * users2[u] for u in common_users)
    return total / len(common_users)


# ---------------------------------------------------
# Recommend top similar movies (Content-based)
# ---------------------------------------------------
def recommend_similar(movie_id, top_n=10):
    results = []

    for other in movie_ratings:
        if other == movie_id:
            continue

        score = movie_similarity(movie_id, other)
        results.append((other, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ---------------------------------------------------
# Recommend movies for a specific user
# ---------------------------------------------------
def recommend_for_user(user_id, top_n=10):
    if user_id not in user_ratings:
        return []

    seen = set(user_ratings[user_id].keys())
    scores = defaultdict(float)

    for movie in seen:
        for other_movie in movie_ratings:
            if other_movie in seen:
                continue
            scores[other_movie] += movie_similarity(movie, other_movie)

    final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return final[:top_n]