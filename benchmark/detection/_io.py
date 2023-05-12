import os
from loguru import logger
import pandas as pd


def write_results_as_cache(
        data: pd.DataFrame, data_name: str, method: str, run: int
):
    result_dir = f"./cache/detection_result/{data_name}/{method}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data.to_csv(os.path.join(result_dir, f"{run}.csv"), index=False)
    logger.opt(colors=True).info(f"<magenta>{method}</magenta> detection results have been cached.")


def read_results_from_cache(data_name: str, method: str, run: int):
    data_dir = f"./cache/detection_result/{data_name}/{method}/{run}.csv"

    if os.path.exists(data_dir):
        logger.opt(colors=True).info(f"Loading cached <magenta>{method}</magenta> detection results...")
        return pd.read_csv(data_dir)
    else:
        return None
