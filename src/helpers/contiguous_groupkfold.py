import numpy as np
from sklearn.model_selection._split import _BaseKFold

class ContiguousGroupKFold(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, random_state = None, shuffle = False)

    def split(self, X, y = None, groups = None):
        if groups is None:
            raise ValueError("The 'groups' parameter is required for ContiguousGroupKFold.")

        # get unique groups
        unique_groups = sorted(np.unique(groups))
        n_groups = len(unique_groups)

        # ensure n_splits is less than or equal to num_groups
        if self.n_splits > n_groups:
            raise ValueError(f"Cannot have number of splits = {self.n_splits} greater than the number of groups = {n_groups}.")

        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[:n_groups % self.n_splits] += 1

        group_starts = np.cumsum(np.insert(fold_sizes, 0, 0))[:-1]

        for i in range(self.n_splits):
            test_start = group_starts[i]
            test_end = test_start + fold_sizes[i]
            test_groups = unique_groups[test_start:test_end]

            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask

            yield np.where(train_mask)[0], np.where(test_mask)[0]
