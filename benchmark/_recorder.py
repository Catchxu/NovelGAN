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
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['Dataset', 'Method', 'Run'])

    records = pd.DataFrame(
        np.full((len(row_tuples), len(metrics)), fill_value=np.nan, dtype=float),
        index=row_index, columns=metrics
    )
    return records


def store_metrics_to_records(
        records: pd.DataFrame,
        metric: Literal['ARI', 'NMI'],
        value: float,
        data_name: str,
        method: Union[str, Callable],
        run: int
):
    records.loc[(data_name, method, run), metric] = value


def write_records(records: pd.DataFrame):
    record_name = f"{datetime.now().strftime('%Y-%m %H_%M_%S')}"
    writer = pd.ExcelWriter(f'{record_name}.xlsx')
    records.to_excel(writer, index=True)
    writer.close()
    logger.info(f"records have been saved into './{record_name}.xlsx'.")
