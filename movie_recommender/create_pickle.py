# imports
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
import time

# Progress indicator
start_time = time.perf_counter()
print('Loading and preparing ml_data')

# load ratings, drop timestamp
ratings = pd.read_csv("ml_data/ratings.csv").drop("timestamp", axis=1)

# load genres, remove rows without genre and remove separator symbol
movies = pd.read_csv("ml_data/movies.csv")

# load tags and remove Null values
tags = pd.read_csv("ml_data/tags.csv").drop("timestamp", axis=1)
tags = tags.loc[tags["tag"].isnull() != True]

# Remove duplicate movie titles
# get dict of movieIDs and titles
titles = movies.set_index("movieId").to_dict()
titles = titles["title"]

# flip keys & values to see which title is double
flipped = {}
for key, value in titles.items():
    if value not in flipped:
        flipped[value] = [key]
    else:
        flipped[value].append(key)

# filter duplicate items
duplicates = {}
for key, value in flipped.items():
    if len(value) > 1:
        duplicates[key] = [value]

# remove duplicates from ratings dataframe
for key, value in duplicates.items():
    movies.drop(movies.loc[movies["movieId"] == duplicates[key][0][1]].index, inplace=True)

# Adjust movie genres
movies = movies.loc[movies["genres"] != "(no genres listed)"]
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Join all tags for each movie together and join with genres in one column
# To not exceed my RAM I'm only including movies that received more than 10 ratings
tags_final = tags.groupby('movieId').filter(lambda x: len(x) > 10).reset_index().drop("index", axis=1)
tags_final = tags_final.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x))
tags_final = pd.merge(movies, tags_final, on="movieId", how="inner").fillna("")
tags_final["tags"] = tags_final["genres"] + " " + tags_final["tag"]
tags_final = tags_final[["movieId", "title", "tags"]]
tags_final = pd.merge(ratings, tags_final, on="movieId", how="inner")
tags_final = tags_final.drop_duplicates("movieId").set_index("movieId")

# Create ratings matrix R
ratings_final = pd.merge(tags_final["title"], ratings, on="movieId", how="left")
ratings_final = ratings_final[["userId", "movieId", "title", "rating"]]
print("Shape of ratings: ", ratings_final.shape)

print("Percentage kept of Users: " + str((len(ratings_final["userId"].unique()) / len(ratings_final["userId"].unique())) * 100))
print("Percentage kept of Movies: " + str((len(ratings_final["movieId"].unique()) / len(ratings["movieId"].unique())) * 100))

# transform into user-item matrix
R = ratings_final.pivot(index="movieId", columns="userId", values="rating")
print("Shape of R: ", R.shape)

# fill NaN with Zero (split in five executions to not exceed RAM)
R[:int(len(R)/5)] = R[:int(len(R)/5)].fillna(0)
R[int(len(R)/5):2*int(len(R)/5)] = R[int(len(R)/5):2*int(len(R)/5)].fillna(0)
R[2*int(len(R)/5):3*int(len(R)/5)] = R[2*int(len(R)/5):3*int(len(R)/5)].fillna(0)
R[3*int(len(R)/5):4*int(len(R)/5)] = R[3*int(len(R)/5):4*int(len(R)/5)].fillna(0)
R[4*int(len(R)/5):] = R[4*int(len(R)/5):].fillna(0)

# Delete dataframes to free memory
del ratings, movies, tags, titles, flipped, duplicates, ratings_final

# Progress indicator
print('Calculating collaborative filter model. Time passed in seconds: ', time.perf_counter() - start_time)

# NMF for collaborative filtering
# train model
collab_model = NMF(n_components=60, init="nndsvd")
transformed_R = collab_model.fit_transform(R)
collab_matrix = pd.DataFrame(transformed_R, index=R.index)

# Export collab_matrix
pickle.dump(collab_matrix, open('binaries/collab_matrix', 'wb'))
del R, collab_model, transformed_R, collab_matrix

# Progress indicator
print('Calculating content-based filter model. Time passed in seconds: ', time.perf_counter() - start_time)

# Tfidf vectorization of the tag strings for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(tags_final['tags'])

# Reduce dimensionality with SVD
n = 4000
content_model = TruncatedSVD(n_components=n)
content_matrix = content_model.fit_transform(tfidf_matrix)
content_matrix = pd.DataFrame(content_matrix, index=tags_final.index)

# Export content_matrix
pickle.dump(content_matrix, open('binaries/content_matrix', 'wb'))

# Export movie title dictionary
movie_titles = tags_final["title"].to_dict()
pickle.dump(movie_titles, open('binaries/movie_titles', 'wb'))

# Export movies as json file for flask/flexdatalist input
with open('static/movies.json', 'w', encoding='utf-8') as file:
    tags_final.reset_index()[["movieId", "title"]].to_json(file, force_ascii=False, orient="records")

# Progress indicator
print('Models calculated and binaries exported. Time passed in seconds: ', time.perf_counter() - start_time)