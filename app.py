# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# -------------------------
# CONFIG
# -------------------------
MOVIES_PKL = r"data/movies.pkl"
TAGS_PKL   = r"data/tags.pkl"
RATINGS_PKL= r"data/ratings.pkl"

# TF-IDF parameters (balance quality vs memory)
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

# number of neighbors to return
NUM_NEIGHBORS = 12  # will remove the query itself later

# -------------------------
# STREAMLIT PAGE
# -------------------------
st.set_page_config(page_title="Hybrid Recommender — Content+Tags", layout="wide")
st.title("🎬 Hybrid Movie Recommender — Genres + Tags")
st.markdown(
    "This app finds movies similar to your selected title using *genres + user tags + title* text.\n\n"
    "It uses TF-IDF (sparse) + NearestNeighbors (cosine) — memory safe for large datasets."
)

# -------------------------
# LOAD DATA (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def load_data(movies_path, tags_path, ratings_path):
    movies = pickle.load(open(movies_path, "rb"))
    tags   = pickle.load(open(tags_path, "rb"))
    ratings= pickle.load(open(ratings_path, "rb"))
    return movies, tags, ratings

movies, tags, ratings = load_data(MOVIES_PKL, TAGS_PKL, RATINGS_PKL)

# -------------------------
# PREPARE TEXT FIELD (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def prepare_text(movies_df: pd.DataFrame, tags_df: pd.DataFrame):
    df = movies_df.copy()

    # Ensure columns exist
    if "title" not in df.columns or "genres" not in df.columns or "movieId" not in df.columns:
        raise ValueError("movies.pkl missing expected columns (movieId,title,genres)")

    # Replace pipe with comma (easier tokenization)
    df["genres"] = df["genres"].fillna("").str.replace("|", ", ")

    # Aggregate tags per movie (join tags into one string)
    # tags DataFrame historically has columns: userId, movieId, tag, timestamp
    tags_df = tags_df.copy()
    if "movieId" in tags_df.columns and "tag" in tags_df.columns:
        agg = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(map(str, x))).rename("tags")
    else:
        # If tags not present or different schema, create empty
        agg = pd.Series(index=df["movieId"].values, data="", name="tags")

    # Merge tags into movies
    df = df.merge(agg, left_on="movieId", right_index=True, how="left")
    df["tags"] = df["tags"].fillna("")

    # Create a combined text field
    # We include title (weight it slightly by repeating), genres, tags
    def make_text(row):
        title = row["title"] if pd.notna(row["title"]) else ""
        genres = row["genres"] if pd.notna(row["genres"]) else ""
        tags = row["tags"] if pd.notna(row["tags"]) else ""
        # repeat title to give more importance
        return " ".join([title] * 2 + [genres] + [tags])

    df["combined_text"] = df.apply(make_text, axis=1)
    return df

movies_df = prepare_text(movies, tags)

# -------------------------
# TF-IDF VECTORIZE + NearestNeighbors (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def build_tfidf_and_index(texts, max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE):
    # Use English stop words and a reasonable max_features to control memory
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    X = vectorizer.fit_transform(texts)  # sparse matrix (n_movies x n_features)
    # Fit nearest neighbors on the sparse matrix using brute force cosine
    nn = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    return vectorizer, X, nn

vectorizer, X_tfidf, nn_model = build_tfidf_and_index(movies_df["combined_text"].values)

# -------------------------
# Helper functions
# -------------------------
movieid_to_title = movies_df.set_index("movieId")["title"].to_dict()
title_to_index = pd.Series(movies_df.index.values, index=movies_df["title"].values).to_dict()

def recommend_by_title(title, top_n=10):
    """
    Returns list of (movie_title, score) for top_n similar movies to given title
    """
    if title not in title_to_index:
        return []
    idx = title_to_index[title]
    x = vectorizer.transform([movies_df.at[idx, "combined_text"]])  # 1 x n_features sparse
    distances, indices = nn_model.kneighbors(x, n_neighbors=NUM_NEIGHBORS, return_distance=True)
    distances = distances.flatten()
    indices = indices.flatten()
    # build results, skip the first result if it's the same movie (distance ~ 0)
    res = []
    for dist, ind in zip(distances, indices):
        movie_id = movies_df.at[ind, "movieId"]
        movie_title = movies_df.at[ind, "title"]
        res.append((movie_title, 1.0 - float(dist)))  # convert cosine distance -> similarity (approx)
    # Remove the query movie itself if present and limit to top_n
    filtered = [r for r in res if r[0] != title]
    return filtered[:top_n]

# -------------------------
# STREAMLIT UI
# -------------------------
st.sidebar.header("Options")
num_display = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)
query_box = st.sidebar.text_input("Search movie title (exact or partial)", "")

st.subheader("Select a movie or search by name")
# Provide a searchable selectbox-like experience by filtering movie list
if query_box.strip():
    # case-insensitive substring match
    matches = movies_df[movies_df["title"].str.contains(query_box, case=False, na=False)]
    options = matches["title"].tolist()
    if not options:
        st.warning("No titles matched your search. Try a different keyword.")
else:
    options = movies_df["title"].tolist()

selected = st.selectbox("Movie", options[:5000] if len(options) > 5000 else options)

if st.button("Get Recommendations"):
    with st.spinner("Computing recommendations..."):
        recs = recommend_by_title(selected, top_n=num_display)
    if not recs:
        st.error("No recommendations found. Try another title.")
    else:
        st.success(f"Top {len(recs)} recommendations for: {selected}")
        for i, (t, score) in enumerate(recs, start=1):
            st.write(f"{i}.** {t} — similarity: {score:.3f}")

st.markdown("---")
st.subheader("Example: Movie Metadata & Sample")
c1, c2 = st.columns([2, 3])

with c1:
    # Show selected movie metadata
    if selected:
        sel_row = movies_df[movies_df["title"] == selected].iloc[0]
        st.write("*Title:*", sel_row["title"])
        st.write("*Genres:*", sel_row["genres"])
        st.write("*Sample Tags (aggregated):*", sel_row["tags"][:400] + ("..." if len(sel_row["tags"])>400 else ""))

with c2:
    st.write("Top 5 movies by rating count (sample)")
    rating_counts = ratings.groupby("movieId").size().reset_index(name="count")
    top5 = rating_counts.sort_values("count", ascending=False).head(5)
    # map ids to titles
    top5["title"] = top5["movieId"].map(movieid_to_title)
    st.table(top5[["title", "count"]].set_index("title"))

st.markdown("---")
st.caption("Notes: TF-IDF max_features set to control memory. If you want stronger results, we can increase max_features or add plot summaries (if available).")