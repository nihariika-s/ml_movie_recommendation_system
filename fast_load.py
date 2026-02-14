import os
import pandas as pd
import pickle

# CSV paths
ratings_csv = "data/ratings_clean.csv"
movies_csv  = "data/movies_clean.csv"
tags_csv    = "data/tags_clean.csv"

# PKL paths
ratings_pkl = "data/ratings.pkl"
movies_pkl  = "data/movies.pkl"
tags_pkl    = "data/tags.pkl"

def fastload(csv_path, pkl_path):
    if os.path.exists(pkl_path):
        print(f"Loaded from cache: {pkl_path}")
        return pickle.load(open(pkl_path, "rb"))
    else:
        df = pd.read_csv(csv_path)
        pickle.dump(df, open(pkl_path, "wb"))
        print(f"CSV loaded & cached as PKL: {pkl_path}")
        return df

ratings = fastload(ratings_csv, ratings_pkl)
movies  = fastload(movies_csv, movies_pkl)
tags    = fastload(tags_csv, tags_pkl)

print("Fast loading complete!")