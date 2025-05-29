import numpy as np
import pandas as pd

from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX

MAX_ROWS = 1_000_000


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", MAX_ROWS)
    no_of_columns = kwargs.get("no_of_columns", None)
    if not nrows:
        nrows = MAX_ROWS
    attr1 = np.random.normal(0, 20, size=nrows)
    attr2 = np.random.normal(0, 20, size=nrows)
    attr3 = attr1 + attr2
    attr4 = attr1 * 2.5 + 5
    attr5 = attr2 * 2.5 + 5

    # Stack the attributes into a 2D array
    data = np.column_stack((attr1, attr2, attr3, attr4, attr5))

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 6)])
    print(df.shape)

    if no_of_columns:
        df = df.iloc[:, :no_of_columns]

    return df


if __name__ == "__main__":
    df, _ = generate_dataset(no_of_columns=3)
    print(df.head())
