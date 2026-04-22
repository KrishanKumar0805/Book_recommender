import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from src.data_loader import load_data
from src.feature_engineering import build_book_profiles, build_user_profiles
from src.collaborative_filter import ALSRecommender
from src.content_filter import ContentRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import leave_one_out_split, evaluate_batch

DATA_DIR = "data/"


def main():
    # 1. Load data
    chapters, interactions, user_book = load_data(
        DATA_DIR + "chapters.csv",
        DATA_DIR + "interactions.csv",
    )

    # 2. Build features
    print("\nBuilding book profiles...")
    book_profiles = build_book_profiles(chapters)

    print("Building user profiles...")
    user_profiles = build_user_profiles(user_book, book_profiles)

    # 3. Split
    print("\nSplitting train / test...")
    train_df, test_dict = leave_one_out_split(user_book)
    print(f"Train: {len(train_df):,} rows | Test users: {len(test_dict):,}")

    # 4. Train ALS
    cf = ALSRecommender(factors=64, iterations=20, regularization=0.1)
    cf.fit(train_df)

    # 5. Content model
    cb = ContentRecommender(book_profiles, user_profiles)

    # 6. Popularity fallback
    popularity = (
        train_df.groupby("book_id")["interaction_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # 7. Hybrid model
    hybrid = HybridRecommender(cf, cb, alpha=0.7, cold_start_threshold=3)

    # 8. Evaluate
    print("\nEvaluating...")
    results = evaluate_batch(cf, test_dict, k=10)
    print(f"\n{'=' * 40}")
    print(f"  HR@10 : {results['HR@K']:.4f}")
    print(f"  MRR   : {results['MRR']:.4f}")
    print(f"  Users : {results['N']:,}")
    print(f"{'=' * 40}")

    # 9. Demo prediction
    demo_user = list(test_dict.keys())[0]
    recs = hybrid.recommend(demo_user, train_df, popularity, n=10)
    true_book = test_dict[demo_user]
    print(f"\nDemo — User : {demo_user}")
    print(f"  True book : {true_book}")
    print(f"  Top-10    : {recs}")
    print(f"  Hit       : {'YES ✓' if true_book in recs else 'NO ✗'}")


if __name__ == "__main__":
    main()