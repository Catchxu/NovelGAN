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

        adata = reader(data_name, data_props, preprocess)

        if preprocess:
            console.print(f"Before QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
            adata = process(adata)
            console.print(f"After QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
        else:
            console.print(f"Skip preprocessing [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")

        adata = feature_select(adata, n_feature=3000)

        # save the cache
        write_adata_as_cache(adata, data_name)

    return adata


def reader(data_name: str,
           data_props: Dict[str, Union[os.PathLike, str]],
           preprocess: bool = True):
    if data_name == 'PBMC(SLE)':
        data1_path = os.path.join(data_props['data_path'], 'pbmcA')
        data2_path = os.path.join(data_props['data_path'], 'pbmcB')
        data3_path = os.path.join(data_props['data_path'], 'pbmcC')

        # read the single-cell data and preprocess
        adata1 = sc.read_10x_mtx(data1_path, var_names='gene_symbols')
        adata2 = sc.read_10x_mtx(data2_path, var_names='gene_symbols')
        adata3 = sc.read_10x_mtx(data3_path, var_names='gene_symbols')
        adatas = [adata1, adata2, adata3]
        adata = ad.concat(adatas, merge='same')
        info = pd.read_csv(data_props['info_path'], sep='\t')
        adata.obs = info.loc[adata.obs.index, ['tsne1', 'tsne2', 'cell.type']]
        adata = adata[~adata.obs['cell.type'].isna(), :]

    # store data name
    adata.uns['data_name'] = data_name

    adata.X = adata.X.todense()

    return adata


def process(adata: ad.AnnData):
    make_unique(adata)
    clean_var_names(adata)

    # quality control
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]

    # store to adata.raw
    adata.raw = adata
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata, base=e)

    # clear unnecessary information
    info = adata.obs.loc[:, ['tsne1', 'tsne2', 'cell.type']]
    adata.obs = info
    info = adata.var.loc[:, ['gene_ids']]
    adata.var = info

    return adata
