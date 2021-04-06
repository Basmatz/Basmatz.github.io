# Imports
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

# load pickle binaries
content_matrix = pickle.load(open('binaries/content_matrix', 'rb'))
collab_matrix = pickle.load(open('binaries/collab_matrix', 'rb'))
movie_titles = pickle.load(open('binaries/movie_titles', 'rb'))

# calculate content-based filtering score
def get_content_scores(movie_list):
    #Dot product score
    score_dot = np.dot(content_matrix, content_matrix.loc[movie_list].T)
    score_dot = pd.DataFrame(score_dot, index=content_matrix.index)
    score_dot["title"] = score_dot.index.map(movie_titles)
    score_dot["sum"] = score_dot.sum(axis=1)
    score_dot["sum"] = minmax_scale(score_dot["sum"], feature_range=(0, 1))

    #Cosine similarity score
    score_cos = cosine_similarity(content_matrix, content_matrix.loc[movie_list])
    score_cos = pd.DataFrame(score_cos, index=content_matrix.index)
    score_cos["title"] = score_cos.index.map(movie_titles)
    score_cos["sum"] = score_cos[list(range(len(movie_list)))].sum(axis=1)
    score_cos["sum"] = minmax_scale(score_cos["sum"], feature_range=(0, 1))

    #Combining both scores with equal weight
    content_based = score_dot.join(score_cos, on="movieId", rsuffix="_cos")
    content_based["content_sum"] = content_based[["sum", "sum_cos"]].sum(1)
    content_based["content_sum"] = minmax_scale(content_based["content_sum"], feature_range=(0, 1))
    content_based.sort_values("content_sum", ascending=False, inplace=True)
    content_based = content_based[["title", "content_sum"]].drop(movie_list)

    return content_based

# calculate collaborative filtering score
def get_collaborative_scores(movie_list):
    #Dot product score
    score_dot = np.dot(collab_matrix, collab_matrix.loc[movie_list].T)
    score_dot = pd.DataFrame(score_dot, index=collab_matrix.index)
    score_dot["title"] = score_dot.index.map(movie_titles)
    score_dot["sum"] = score_dot[list(range(len(movie_list)))].sum(axis=1)
    score_dot["sum"] = minmax_scale(score_dot["sum"], feature_range=(0, 1))

    #Cosine similarity score
    score_cos = cosine_similarity(collab_matrix, collab_matrix.loc[movie_list])
    score_cos = pd.DataFrame(score_cos, index=collab_matrix.index)
    score_cos["title"] = score_cos.index.map(movie_titles)
    score_cos["sum"] = score_cos[list(range(len(movie_list)))].sum(axis=1)
    score_cos["sum"] = minmax_scale(score_cos["sum"], feature_range=(0, 1))

    #Combining both scores with lower maximum scale to put less weight on collaborative filtering.
    collaborative_filter = score_dot.join(score_cos, on="movieId", rsuffix="_cos")
    collaborative_filter["colab_sum"] = collaborative_filter[["sum", "sum_cos"]].sum(1)
    collaborative_filter["colab_sum"] = minmax_scale(collaborative_filter["colab_sum"], feature_range=(0, 0.4))
    collaborative_filter.sort_values("colab_sum", ascending=False, inplace=True)
    collaborative_filter = collaborative_filter[["title", "colab_sum"]].drop(movie_list)

    return collaborative_filter

# calculate combined score of collaborative and content-based filtering.
def get_combined_scores(movie_list):
    content_filter = get_content_scores(movie_list)
    collaborative_filter = get_collaborative_scores(movie_list)
    final_combined = pd.concat([content_filter, collaborative_filter["colab_sum"]], axis=1, join="inner")
    final_combined["final_sum"] = final_combined[["content_sum", "colab_sum"]].sum(1)
    final_combined.sort_values("final_sum", ascending=False, inplace=True)

    return collaborative_filter.head(10).values#final_combined["title"].head(10).values