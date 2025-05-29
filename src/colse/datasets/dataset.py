
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    data: pd.DataFrame
    name: str
    continuous_columns: list
    categorical_columns: list


current_dir_path = Path(__file__).parent


def get_dataset_files(dataset_name):
    if dataset_name == "forest":
        return current_dir_path.joinpath("../../data/forest/forest.csv").resolve()
    if "employee_dataset" in dataset_name:
        return current_dir_path.joinpath(f"../../data/employee/{dataset_name}.csv").resolve()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
