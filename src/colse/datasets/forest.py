from pathlib import Path

import pandas as pd

from cest.datasets.dataset import Dataset
from colse.data_path import get_data_path

all_cols = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
    'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]


def load_forest_data(dataset_path='forest.csv', no_of_columns=None, no_of_rows=None):
    db_path = get_data_path() / f"forest/{dataset_path}"
    df = pd.read_csv(db_path, index_col=None)

    if no_of_columns is None:
        no_of_columns = len(all_cols)

    if no_of_columns > len(all_cols):
        raise ValueError(f"Number of columns should be less than {len(all_cols)}")

    if no_of_rows is not None:
        df = df.sample(n=no_of_rows, random_state=1)

    return Dataset(
        data=df.iloc[:, :no_of_columns],
        name=db_path.stem,
        continuous_columns=all_cols[:no_of_columns],
        categorical_columns=[]
    )
