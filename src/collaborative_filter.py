import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors=64, iterations=20, regularization=0.1):
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=False,
        )
        self.user_map = {}
        self.book_map = {}
        self.inv_book_map = {}
        self.matrix = None

    def fit(self, user_book_df):
        users = user_book_df["user_id"].unique()
        books = user_book_df["book_id"].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.book_map = {b: i for i, b in enumerate(books)}
        self.inv_book_map = {i: b for b, i in self.book_map.items()}

        rows = user_book_df["user_id"].map(self.user_map)
        cols = user_book_df["book_id"].map(self.book_map)
        data = user_book_df["interaction_count"].astype(float)

        self.matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(books))
        )
        print("Fitting ALS model...")
        self.model.fit(self.matrix)
        print("ALS training complete.")

    def recommend(self, user_id: str, n: int = 200) -> list:
        if user_id not in self.user_map:
            return []
        uid = self.user_map[user_id]
        ids, scores = self.model.recommend(
            uid, self.matrix[uid], N=n, filter_already_liked_items=True
        )
        return [(self.inv_book_map[i], float(s)) for i, s in zip(ids, scores)]