import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, book_profiles: pd.DataFrame, user_profiles: dict):
        self.book_profiles = book_profiles
        self.user_profiles = user_profiles

    def recommend(self, user_id: str, read_books: set, n: int = 200) -> list:
        if user_id not in self.user_profiles:
            return []
        user_vec = self.user_profiles[user_id].reshape(1, -1)
        unseen = self.book_profiles[~self.book_profiles.index.isin(read_books)]
        if unseen.empty:
            return []
        sims = cosine_similarity(user_vec, unseen.values)[0]
        top_idx = sims.argsort()[::-1][:n]
        return [(unseen.index[i], float(sims[i])) for i in top_idx]