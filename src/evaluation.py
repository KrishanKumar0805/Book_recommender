import pandas as pd
import numpy as np


def leave_one_out_split(user_book_df: pd.DataFrame, seed: int = 42):
    train_parts, test = [], {}
    for user, group in user_book_df.groupby("user_id"):
        if len(group) < 2:
            train_parts.append(group)
            continue
        held = group.sample(1, random_state=seed)
        test[user] = int(held["book_id"].values[0])
        train_parts.append(group.drop(held.index))
    return pd.concat(train_parts, ignore_index=True), test


def evaluate_batch(cf_model, test_dict: dict, k: int = 10) -> dict:
    """
    Fast batch evaluation using ALS directly.
    Skips the slow per-user hybrid loop — uses CF model only for speed.
    """
    print(f"  Running batch ALS recommendations for {len(test_dict):,} users...")

    # Only evaluate users the CF model knows about
    known_users = {u: cf_model.user_map[u] for u in test_dict if u in cf_model.user_map}
    user_ids = list(known_users.keys())
    user_indices = list(known_users.values())

    print(f"  Known users in CF model: {len(user_ids):,}")

    # Batch recommend — much faster than one-by-one
    batch_size = 5000
    hits, reciprocal_ranks = 0, []

    for start in range(0, len(user_indices), batch_size):
        end = min(start + batch_size, len(user_indices))
        batch_uids = user_indices[start:end]
        batch_user_ids = user_ids[start:end]

        # Get user vectors for this batch
        user_vecs = cf_model.matrix[batch_uids]

        # Batch recommend
        batch_ids, batch_scores = cf_model.model.recommend(
            batch_uids,
            user_vecs,
            N=k,
            filter_already_liked_items=True,
        )

        for i, user in enumerate(batch_user_ids):
            true_book = test_dict[user]
            rec_book_ids = [cf_model.inv_book_map[idx] for idx in batch_ids[i]]

            if true_book in rec_book_ids:
                hits += 1
                rank = rec_book_ids.index(true_book) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        print(f"  Progress: {end:,} / {len(user_indices):,}", end="\r")

    total = len(reciprocal_ranks)
    print(f"\n  Done evaluating {total:,} users.")
    return {
        "HR@K": hits / total,
        "MRR":  sum(reciprocal_ranks) / total,
        "K":    k,
        "N":    total,
    }
    