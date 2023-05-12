import random
import numpy as np
import anndata as ad
from typing import Literal
import warnings
warnings.filterwarnings('ignore')


def split(adata: ad.AnnData,
          remove_cell: str,
          random_state: int,
          train_size: float = 0.7):
    np.random.seed(random_state)
    random.seed(random_state)

    shuffle_id = np.random.permutation(adata.n_obs)
    shuffle_id = adata.obs.index[shuffle_id]

    train_size = int(adata.n_obs * train_size)
    train_id = shuffle_id[:train_size]
    test_id = shuffle_id[train_size:]
    train = adata[train_id, :]
    test = adata[test_id, :]

    train = train[~train.obs['cell.type'].str.contains(remove_cell), :]

    return train, test