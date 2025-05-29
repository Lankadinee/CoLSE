from pathlib import Path

import numpy as np
import pandas as pd

from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX

"""Custom dataset"""

TYPE = 0


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", 500_000)
    value_range = 200

    if TYPE == 0:
        attr1 = np.linspace(
            -value_range / 2, value_range / 2, nrows, dtype=float
        ) + np.random.normal(0, 1, size=nrows)
        att2_1 = np.linspace(
            -value_range / 2, value_range / 2, int(np.floor(nrows / 2)), dtype=float
        )
        att2_2 = np.linspace(
            value_range / 2, -value_range / 2, int(np.ceil(nrows / 2)), dtype=float
        )
        attr2 = np.concatenate((att2_1, att2_2)) + np.random.normal(0, 1, size=nrows)
    elif TYPE == 1:
        attr1 = np.linspace(
            -value_range / 2, value_range / 2, nrows, dtype=float
        ) + np.random.normal(0, 1, size=nrows)
        attr2 = np.linspace(
            -value_range / 2, value_range / 2, nrows, dtype=float
        ) + np.random.normal(0, 1, size=nrows)
    else:
        attr1 = np.random.normal(0, 1, size=nrows)
        attr2 = np.random.normal(0, 1, size=nrows)

    # attr3 = attr1 + attr2
    # attr4 = attr1 * 2.5 + 5
    # attr5 = attr2 * 2.5 + 5
    # attr6 = attr1 ** 2 / 100 + attr5
    # attr7 = attr1 + attr4 + np.random.normal(0, 2, size=nrows)
    # attr8 = attr3 + attr5 + attr6 + attr7
    # attr9 = attr1 + attr2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])

    # Stack the attributes into a 2D array
    # data = np.column_stack((attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9))
    data = np.column_stack((attr1, attr2))

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 3)])
    print(df.shape)
    return df


if __name__ == "__main__":
    generate_dataset()
