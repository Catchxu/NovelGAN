import os
from math import e
from typing import Dict
from typing import Union

import anndata as ad
import scanpy as sc
from loguru import logger

from ._feature import feature_select
from ._io import write_adata_as_cache, read_adata_from_cache
from ._utils import make_unique, clean_var_names, clear_info
from .._utils import console


def load_data(data_name: str,
              data_props: Dict[str, Union[os.PathLike, str]],
              preprocess: bool = True):
    console.rule('[bold red]' + data_name)
    logger.info("Reading adata...")

    if data_name == 'PBMC':
        train = reader(data_props, 'train')
        test1 = reader(data_props, 'test1')
        test2 = reader(data_props, 'test2')

        adatas = [train, test1, test2]
        adatas = ad.concat(adatas, merge='same')
        genes = feature_select(adatas, n_feature=3000)
        train = train[:, genes]
        test1 = test1[:, genes]
        test2 = test2[:, genes]

        train, test1, test2 = remove(train, test1, test2)

        return train, test1, test2


def reader(data_props: Dict[str, Union[os.PathLike, str]],
           batch_type: str,
           preprocess: bool = True):
    adata = read_adata_from_cache(batch_type)
    if isinstance(adata, ad.AnnData):
        console.print(f"Using cached adata [bold red]{batch_type}[/bold red] and skip preprocessing. \n"
                      f"Shape: [yellow]{adata.n_obs}[/yellow] cells, [yellow]{adata.n_vars}[/yellow] genes.")
    else:
        logger.opt(colors=True).info(
            f"No cache for <magenta>{batch_type}</magenta>. Trying to read data from the given path in config..."
        )
        console.print(f"Reading adata [bold red]{batch_type}[/bold red]")
        adata = sc.read(data_props[batch_type], var_names='gene_symbols')

        if preprocess:
            console.print(f"Before QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
            adata = process(adata)
            console.print(f"After QC: [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")
        else:
            console.print(f"Skip preprocessing [yellow]{adata.n_obs}[/yellow] cells and [yellow]{adata.n_vars}[/yellow] genes.")

        adata.X = adata.X.toarray()
        write_adata_as_cache(adata, batch_type)

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
    adata.raw = clear_info(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata, base=e)

    adata = clear_info(adata)
    return adata


def remove(train: ad.AnnData, test1: ad.AnnData, test2: ad.AnnData):
    type1 = 'Monocytes'
    type2 = 'B cells'

    train = train[~train.obs['cell.type'].str.contains(type1), :]
    train = train[~train.obs['cell.type'].str.contains(type2), :]
    train.obs['label'] = 0

    # B cells is novel cell type and Monocytes have been removed
    test1 = test1[~test1.obs['cell.type'].str.contains(type1), :]
    test1.obs['label'] = 0
    test1.obs.loc[test1.obs['cell.type'].str.contains(type2), 'label'] = 1

    # Monocytes is novel cell type and B cells have been removed
    test2 = test2[~test2.obs['cell.type'].str.contains(type2), :]
    test2.obs['label'] = 0
    test2.obs.loc[test2.obs['cell.type'].str.contains(type1), 'label'] = 1

    return train, test1, test2