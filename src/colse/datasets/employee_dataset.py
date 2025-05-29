from pathlib import Path

import pandas as pd

from cest.datasets.dataset import Dataset, get_dataset_files

all_cols = [
    'first_name', 'last_name', 'gender', 'age', 'salary', 'position', 'ID'
]
continuous_cols = ['age', 'salary']


def load_employee_data(dataset_path=get_dataset_files('employee_dataset'), no_of_rows=None):
    """column_names: list of column names to be loaded from the dataset"""
    db_path = Path(dataset_path)
    df = pd.read_csv(db_path, index_col=None)

    # if column_names is None:
    #     column_names = all_cols
    #
    # if len(column_names) > len(all_cols):
    #     raise ValueError(f"Number of columns should be less than {len(all_cols)}")

    if no_of_rows is not None:
        df = df.sample(n=no_of_rows, random_state=1)

    return Dataset(
        data=df,
        name=db_path.stem,
        continuous_columns=continuous_cols,
        categorical_columns=[col for col in all_cols if col not in continuous_cols]
    )
