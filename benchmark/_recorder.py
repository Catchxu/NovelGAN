import os
from datetime import datetime
from itertools import product
from typing import Dict, Union, List, Literal, Callable

import numpy as np
import pandas as pd
from loguru import logger


def create_records(
        data_cfg: Dict[str, Dict[str, Union[os.PathLike, str, List, Dict[str, Union[List, str]]]]],
        method_cfg: Dict[str, int],
        metrics: List[Literal['ARI', 'NMI']]
):
    row_tuples = [
        tup
        for method, n_runs in method_cfg.items()
        for tup in product(data_cfg.keys(), (method.__name__ if callable(method) else method,), range(n_runs))
    ]
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['dataset', 'method', 'run'])

    single_record = pd.DataFrame(
        np.full((len(row_tuples), 1), fill_value=np.nan, dtype=float),
        index=row_index, columns=['value']
    )
    return {metric: single_record.copy() for metric in metrics}


def store_metrics_to_records(
        records: Dict[str, pd.DataFrame],
        metric: Literal['ARI', 'NMI'],
        value: float,
        data_name: str,
        method: Union[str, Callable],
        run: int
):
    if callable(method):
        method = method.__name__
    records[metric].loc[(data_name, method, run), 1] = value


def write_records(records: Dict[str, pd.DataFrame]):
    record_name = f"{datetime.now().strftime('%Y-%m %H_%M_%S')}"
    writer = pd.ExcelWriter(f'{record_name}.xlsx')
    for metric, record in records.items():
        record.to_excel(writer, sheet_name=metric, index=True)
    writer.close()
    logger.info(f"records have been saved into './{record_name}.xlsx'.")
