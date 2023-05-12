import os
from math import e
from typing import Dict
from typing import Union

import pandas as pd
import anndata as ad
import scanpy as sc
from loguru import logger

from ._feature import feature_select
from ._io import write_adata_as_cache, read_adata_from_cache
from ._preprocess import make_unique, clean_var_names
from .._utils import console


def load_data(data_name: str,
              data_props: Dict[str, Union[os.PathLike, str]],
              preprocess: bool = True):
    name_list = data_name.split(sep='+')
    data_name1 = name_list[0]
    data_name2 = name_list[1]

    data1_path = data_props['data1_path']
    data2_path = data_props['data2_path']
    info_path = data_props['info_path']

    adata1 = reader(data_name1, data1_path, info_path, preprocess)
    adata2 = reader(data_name2, data2_path, info_path, preprocess)
    adata1, adata2 = feature_select(adata1, adata2, n_feature=3000)

    return adata1, adata2


def reader(data_name: str,
           data_path: Union[os.PathLike, str],
           info_path: Union[os.PathLike, str],
           preprocess: bool = True):
    console.rule('[bold red]' + data_name)
    logger.info("Reading adata...")

    adata = read_adata_from_cache(data_name)
    if isinstance(adata, ad.AnnData):
        console.print(f"Using cached adata and skip preprocessing. \n"
                      f"Shape: [yellow]{adata.n_obs}[/yellow] cells, [yellow]{adata.n_vars}[/yellow] genes.")
    else:
        logger.opt(colors=True).info(
            f"No cache for <magenta>{data_name}</magenta>. Trying to read data from the given path in config..."
        )

        # read the single-cell data and preprocess
        adata = sc.read_10x_mtx(data_path, var_names='gene_symbols')
        info = pd.read_csv(info_path, sep='\t')
        adata.obs = info.loc[adata.obs.index, ['tsne1', 'tsne2', 'cell.type']]
        adata = adata[~adata.obs['cell.type'].isna(), :]

        if preprocess:
            adata = process(adata)

        console.print(f"Having read adata and finished the preprocess. \n"
                      f"Shape: [yellow]{adata.n_obs}[/yellow] cells, [yellow]{adata.n_vars}[/yellow] genes.")

        # save the cache
        write_adata_as_cache(adata, data_name)
        
        adata.X = adata.X.todense()

    return adata


def process(adata: ad.AnnData):
    make_unique(adata)
    clean_var_names(adata)

    # quality control
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]

    # store to adata.raw
    adata.raw = adata

    # normalization
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata, base=e)
    # sc.pp.scale(adata, max_value=10)

    # remove non-essential information
    info = adata.obs.loc[:, ['tsne1', 'tsne2', 'cell.type']]
    adata.obs = info
    info = adata.var.loc[:, ['gene_ids']]
    adata.var = info

    return adata
