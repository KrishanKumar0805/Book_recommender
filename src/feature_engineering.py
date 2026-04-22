import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

GENRES = [
    "Adventure", "Crime", "Dystopian", "Fantasy", "Graphic Novel",
    "Historical Fiction", "Horror", "Humor", "Literary Fiction",
    "Mystery", "Paranormal", "Romance", "Science Fiction",
    "Thriller", "Young Adult",
]


def build_book_profiles(chapters: pd.DataFrame) -> pd.DataFrame:
    book_genres = (
        chapters.groupby("book_id")["genre_list"]
        .apply(lambda x: list({g for genres in x for g in genres}))
    )
    mlb = MultiLabelBinarizer(classes=GENRES)
    matrix = mlb.fit_transform(book_genres)
    return pd.DataFrame(matrix, index=book_genres.index, columns=GENRES)


def build_user_profiles(user_book: pd.DataFrame, book_profiles: pd.DataFrame) -> dict:
    user_profiles = {}
    for user, group in user_book.groupby("user_id"):
        valid_books = [b for b in group["book_id"] if b in book_profiles.index]
        if not valid_books:
            continue
        vecs = book_profiles.loc[valid_books].values
        weights = (
            group.set_index("book_id")
            .loc[valid_books, "interaction_count"]
            .values
            .astype(float)
        )
        user_profiles[user] = np.average(vecs, axis=0, weights=weights)
    return user_profiles