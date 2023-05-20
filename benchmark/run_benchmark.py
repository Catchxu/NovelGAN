import os
from typing import List, Dict, Union, Literal, Optional

from loguru import logger

# from ._metrics import compute_clustering_metrics
from ._recorder import create_records, write_records, store_metrics_to_records
from ._utils import rm_cache, set_logger
from ._split import split
from .dataset import load_data
from .detection import generally_detect


@logger.catch
def cell_detect_bench(
        data_cfg: Dict[str, Dict[str, Union[os.PathLike, str, List, Dict[str, Union[List, str]]]]],
        method_cfg: Dict[str, int],
        metrics: List[Literal['ARI', 'NMI']],
        preprocess: bool = True,
        clean_cache: bool = False,
        verbosity: Literal[0, 1, 2] = 2,
        log_path: Optional[Union[os.PathLike, str]] = None,
        random_state: int = 0
):
    set_logger(verbosity, log_path)
    if clean_cache:
        rm_cache("./cache")
    records = create_records(data_cfg, method_cfg, metrics)

    for data_name, data_props in data_cfg.items():
        adata = load_data(data_name, data_props, preprocess)
        remove_cell = 'Monocytes'
        train, test = split(adata, remove_cell, random_state)
        for method, n_runs in method_cfg.items():
            for run in range(n_runs):
                result = generally_detect(train, test, data_name, method, run, random_state)
                # to be updated
                for metric in metrics:
                    value = 1
                    store_metrics_to_records(records, metric, value, data_name, method, run)

    # store_metrics_to_records
    write_records(records)