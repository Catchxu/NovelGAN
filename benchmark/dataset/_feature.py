import pandas as pd
import anndata as ad
import anndata2ri
from typing import Optional
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr

from ._utils import rpy2_wrapper


@rpy2_wrapper
def feature_select(adata: ad.AnnData, n_feature: int = 3000) -> Optional[pd.DataFrame]:
    genes = adata.var.index
    adata = adata.raw.to_adata()
    adata = adata[:, genes]
    importr("M3Drop")
    globalenv['sce'] = anndata2ri.py2rpy(adata)
    r("""
    norm <- M3DropConvertData(assay(sce, 'X'), is.counts=TRUE)
    DE_genes <- M3DropFeatureSelection(norm, mt_method="fdr", mt_threshold=1, suppress.plot = TRUE)
    """)
    result = r("rownames(DE_genes)")
    result = list(result)
    genes = result[0:n_feature]
    return genes
