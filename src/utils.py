import pandas as pd
import numpy as np


def train_test_split_by_user(df, test_size=0.2, seed=42):

    train_list = []
    test_list = []

    rng = np.random.default_rng(seed)

    for user_id, group in df.groupby("userId"):

        if len(group) < 2:
            train_list.append(group)
            continue

        n_test = max(1, int(len(group) * test_size))

        test_idx = rng.choice(
            group.index,
            size=n_test,
            replace=False
        )

        test = group.loc[test_idx]
        train = group.drop(test_idx)

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df