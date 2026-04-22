import pandas as pd


def load_data(chapters_path: str, interactions_path: str):
    chapters = pd.read_csv(chapters_path)
    interactions = pd.read_csv(interactions_path)

    chapters["genre_list"] = chapters["tags"].str.split("|")

    user_book = (
        interactions
        .groupby(["user_id", "book_id"])
        .size()
        .reset_index(name="interaction_count")
    )

    print(f"Loaded {len(chapters):,} chapters, {len(interactions):,} interactions")
    print(f"Users: {interactions['user_id'].nunique():,} | Books: {interactions['book_id'].nunique():,}")
    return chapters, interactions, user_book