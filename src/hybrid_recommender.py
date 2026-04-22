import pandas as pd
from src.collaborative_filter import ALSRecommender
from src.content_filter import ContentRecommender


class HybridRecommender:
    def __init__(
        self,
        cf_model: ALSRecommender,
        cb_model: ContentRecommender,
        alpha: float = 0.7,
        cold_start_threshold: int = 3,
    ):
        self.cf = cf_model
        self.cb = cb_model
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold

    @staticmethod
    def _normalize(score_dict: dict) -> dict:
        if not score_dict:
            return {}
        mn, mx = min(score_dict.values()), max(score_dict.values())
        if mx == mn:
            return {k: 1.0 for k in score_dict}
        return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

    def recommend(
        self,
        user_id: str,
        user_book_df: pd.DataFrame,
        popularity_fallback: list,
        n: int = 10,
    ) -> list:
        read_books = set(
            user_book_df[user_book_df["user_id"] == user_id]["book_id"]
        )

        if len(read_books) == 0:
            return popularity_fallback[:n]

        cf_recs = self._normalize(dict(self.cf.recommend(user_id, n=200)))
        cb_recs = self._normalize(
            dict(self.cb.recommend(user_id, read_books, n=200))
        )

        alpha = self.alpha if len(read_books) >= self.cold_start_threshold else 0.0

        all_books = set(cf_recs) | set(cb_recs)
        scores = {
            b: alpha * cf_recs.get(b, 0.0) + (1 - alpha) * cb_recs.get(b, 0.0)
            for b in all_books
        }
        return sorted(scores, key=scores.get, reverse=True)[:n]