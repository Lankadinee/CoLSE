from pathlib import Path

import pandas as pd
from loguru import logger

def load_dataframe(file_path: str | Path):
    logger.info(f"Loading dataframe from {file_path}")
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".xlsx":
        df =  pd.read_excel(file_path)
    elif file_path.suffix == ".parquet":
        df =  pd.read_parquet(file_path)
    else:
        raise ValueError(f"File type {file_path.suffix} not supported")
    return df.dropna()


def save_dataframe(df: pd.DataFrame, file_path: str | Path):
    logger.info(f"Saving dataframe to {file_path}")
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_path.suffix == ".xlsx":
        df.to_excel(file_path, index=False)
    elif file_path.suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"File type {file_path.suffix} not supported")
