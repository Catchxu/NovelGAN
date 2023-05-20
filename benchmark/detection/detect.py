import scanpy as sc
import pandas as pd
from loguru import logger
import anndata as ad
import anndata2ri
from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr

from ._io import read_results_from_cache, write_results_as_cache
from NovelGAN import Detect_cell
from ..dataset._utils import rpy2_wrapper
import warnings
warnings.filterwarnings("ignore")


def generally_detect(
    train: ad.AnnData, test: ad.AnnData,
    data_name: str, method: str, run: int, random_state: int
):
    result = read_results_from_cache(data_name, method, run)
    if result is None:
        result = detect_cell_type(train, test, method, random_state=random_state+run)
        write_results_as_cache(result, data_name, method, run)

    return result


def detect_cell_type(
    train: ad.AnnData, test: ad.AnnData, method: str, random_state: int
):
    logger.opt(colors=True).info(f"Running <magenta>{method}</magenta> to detect novel cell type"
                                 f"and random state <yellow>{random_state}</yellow>...")
    if method == 'NovelGAN':
        return detect_NovelGAN(train, test, random_state)
    elif method == 'CAMLU':
        return detect_CAMLU(train, test, random_state)
    elif method == 'scmap-cluster':
        return detect_scmap_cluster(train, test, random_state)
    elif method == 'scPred':
        return detect_scPred(train, test, random_state)
    elif method == 'CHETAH':
        return detect_CHETAH(train, test, random_state)
    else:
        raise NotImplementedError(f"{method} has not been implemented.")


def detect_NovelGAN(train: ad.AnnData, test: ad.AnnData, random_state: int):
    diff = Detect_cell(train.X, test.X, verbose=False)
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': diff}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


@rpy2_wrapper
def detect_CAMLU(train: ad.AnnData, test: ad.AnnData, random_state: int):
    train = train.raw.to_adata()
    test = test.raw.to_adata()
    importr("CAMLU")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    label <- CAMLU(x_train = assay(train,'X'),
                   x_test = assay(test,'X'),
                   ngene=3000, lognormalize=TRUE)
    """)
    pre_label = list(r('label'))
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


@rpy2_wrapper
def detect_scmap_cluster(train: ad.AnnData, test: ad.AnnData, random_state: int):
    train = train.raw.to_adata()
    test = test.raw.to_adata()
    sc.pp.normalize_per_cell(train)
    sc.pp.log1p(train, base=2)
    sc.pp.normalize_per_cell(test)
    sc.pp.log1p(test, base=2)

    importr("scmap")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    logcounts(train) <- assay(train, 'X')
    rowData(train)$feature_symbol <- rownames(train)
    colData(train)$cell_type1 = colData(train)$cell.type
    train <- selectFeatures(train, suppress_plot = TRUE)
    train <- indexCluster(train)

    logcounts(test) <- assay(test, 'X')
    rowData(test)$feature_symbol <- rownames(test)
    scmapCluster_results <- scmapCluster(
      projection = test,
      index_list = list(
        metadata(train)$scmap_cluster_index
      )
    )
    """)

    pre_label = list(r('scmapCluster_results$scmap_cluster_labs'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


@rpy2_wrapper
def detect_scPred(train: ad.AnnData, test: ad.AnnData, random_state: int):
    train = train.raw.to_adata()
    test = test.raw.to_adata()
    importr("scPred")
    importr("Seurat")
    importr("magrittr")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    set.seed(seed)

    cell_type <- colData(train)
    train <- CreateSeuratObject(assay(train, 'X'),
                                cell_metadata=colData(train),
                                feat_metadata=rowData(train))
    train <- train %>%
        NormalizeData() %>%
        FindVariableFeatures() %>%
        ScaleData() %>%
        RunPCA()
    train@meta.data <- data.frame(train@meta.data, cell_type)

    train <- getFeatureSpace(train, 'cell.type')
    train <- trainModel(train)

    test <- CreateSeuratObject(assay(test, 'X'),
                               cell_metadata=colData(test),
                               feat_metadata=rowData(test))
    test <- NormalizeData(test)
    test <- scPredict(test, train)
    """)
    pre_label = list(r('test@meta.data$scpred_prediction'))
    pre_label = [1 if i == 'unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result


@rpy2_wrapper
def detect_CHETAH(train: ad.AnnData, test: ad.AnnData, random_state: int):
    importr("CHETAH")
    globalenv['seed'] = random_state
    globalenv['train'] = anndata2ri.py2rpy(train)
    globalenv['test'] = anndata2ri.py2rpy(test)
    r("""
    colnames(colData(train)) <- 'celltypes'
    test <- CHETAHclassifier(input = test, ref_cells = train)
    """)
    pre_label = list(r('colData(test)$celltype_CHETAH'))
    pre_label = [1 if i == 'Unassigned' else 0 for i in pre_label]
    result = {'cell_type': test.obs['cell.type'].values,
              'label': test.obs['label'].values,
              'diff': pre_label}
    result = pd.DataFrame(result, index=test.obs.index)
    return result