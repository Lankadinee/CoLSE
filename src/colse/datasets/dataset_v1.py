import numpy as np
import pandas as pd

from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX

MAX_ROWS = 1_000_000


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", MAX_ROWS)
    selected_cols = kwargs.get("selected_cols", None)

    if not nrows:
        nrows = MAX_ROWS
    attr1 = np.random.normal(0, 20, size=nrows)
    attr2 = np.random.normal(0, 20, size=nrows)
    attr3 = attr1 + attr2
    attr4 = attr1 * 2.5 + 5
    attr5 = attr2 * 2.5 + 5
    attr6 = attr1**2 / 100 + attr5
    attr7 = attr1 + attr4 + np.random.normal(0, 2, size=nrows)
    attr8 = attr3 + attr5 + attr6 + attr7
    attr9 = attr1 + attr2 + np.random.choice([0, 1], size=nrows, p=[0.8, 0.2])
    attr10 = (
        np.random.randint(10, size=nrows) ** 2
        + np.random.randint(3, size=nrows) * attr4 / 100
    )
    attr11 = attr3 * 2 + attr4 + np.random.normal(0, 3, size=nrows)
    attr12 = np.random.exponential(scale=5, size=nrows)
    attr13 = np.random.gamma(shape=0.5, scale=1, size=nrows)
    attr14 = np.random.exponential(scale=5, size=nrows) + np.random.normal(
        0, 2, size=nrows
    )

    # Stack the attributes into a 2D array
    data = np.column_stack(
        (
            attr1,
            attr2,
            attr3,
            attr4,
            attr5,
            attr6,
            attr7,
            attr8,
            attr9,
            attr10,
            attr11,
            attr12,
            attr13,
            attr14,
        )
    )

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 15)])
    print(df.shape)

    if selected_cols:
        df = df.iloc[:, selected_cols]

    return df
