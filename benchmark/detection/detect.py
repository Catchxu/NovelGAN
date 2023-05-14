import pandas as pd
import anndata as ad
from loguru import logger

from ._io import read_results_from_cache, write_results_as_cache
from NovelGan import Detect_cell


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
    if method == 'NovelGan':
        return detect_NovelGan(train, test, random_state)
    elif method == 'CAMLU':
        return None
    else:
        raise NotImplementedError(f"{method} has not been implemented.")


def detect_NovelGan(train: ad.AnnData, test: ad.AnnData, random_state: int):
    diff = Detect_cell(train.X, test.X, verbose=False)
    result = pd.DataFrame({'cell_type': test.obs['cell.type'], 
                           'label': test.obs['label'],
                           'diff': diff})
    return result

