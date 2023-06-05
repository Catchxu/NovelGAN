import traceback
import anndata2ri
from rpy2.robjects import pandas2ri
import anndata as ad
import numpy as np
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse
from .._utils import HiddenPrints
import re


def subsample(adata: ad.AnnData, n_sampled_obs: int = None, n_sampled_vars: int = None, random_state: int = 0):
    assert n_sampled_obs is not None or n_sampled_vars is not None, \
        "Specify at least one of `n_sampled_obs` and `n_sampled_vars`."
    if n_sampled_obs is not None:
        adata = sc.pp.subsample(adata, n_obs=n_sampled_obs, random_state=random_state, copy=True)
        sc.pp.filter_genes(adata, min_cells=1)  # filter genes with no count

    if n_sampled_vars is not None:
        rng = np.random.default_rng(random_state)
        adata = adata[:, rng.choice(adata.n_vars, size=int(n_sampled_vars), replace=False)].copy()
        sc.pp.filter_cells(adata, min_genes=1)  # filter cells with no count

    logger.debug(f"Shape of sampled adata : {adata.shape}")


def to_dense(adata: ad.AnnData):
    if issparse(adata.X):
        logger.info(f"Found sparse matrix in `adata.X`. Converting to ndarray...")
        adata.X = adata.X.toarray()


def is_normalized(adata: ad.AnnData):
    return not np.allclose(adata.X % 1, 0)


def clean_var_names(adata: ad.AnnData):
    adata.var['original_name'] = adata.var_names
    logger.debug("The original variable names have been saved to `adata.var['original_name']`.")
    gene_names = adata.var_names.to_numpy()
    regex = re.compile(pattern='[-_:+()|]')
    vreplace = np.vectorize(lambda x: regex.sub('.', x), otypes=[str])
    adata.var_names = vreplace(gene_names)


def make_unique(adata: ad.AnnData):
    if adata.obs_names.has_duplicates:
        logger.debug("Observation names have duplicates. Making them unique...")
        adata.obs_names_make_unique(join='.')
    if adata.var_names.has_duplicates:
        logger.debug("Variables names have duplicates. Making them unique...")
        adata.var_names_make_unique(join='.')
    logger.info("Observation names and Variables names are all unique now.")


def rpy2_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            with HiddenPrints():
                anndata2ri.activate()
                pandas2ri.activate()
                res = func(*args, **kwargs)
                pandas2ri.deactivate()
                anndata2ri.deactivate()
                return res
        except:
            traceback.print_exc()
    return wrapper


def clear_info(adata: ad.AnnData):
    # clear unnecessary information
    info = adata.obs.loc[:, ['cell.type']]
    adata.obs = info
    info = adata.var.loc[:, ['gene_ids']]
    adata.var = info
    return adata