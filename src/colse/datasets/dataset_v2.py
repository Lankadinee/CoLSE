from pathlib import Path

import numpy as np
import pandas as pd

from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX

"""Payroll dataset"""


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", 50_000)
    """ Load dataset"""
    current_dir = Path(__file__).parent
    df = pd.read_csv(
        current_dir.joinpath("../../data/payroll/Citywide_Payroll_Data.csv")
    )
    df["Working Years"] = df["Agency Start Date"].apply(
        lambda x: 2023 - int(x.split("/")[-1]) if pd.notna(x) else None
    )

    # Select only the continuous columns
    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)

    nrows = df.shape[0] if nrows is None else nrows

    attr1 = df["Base Salary"].to_numpy()[:nrows]
    attr2 = df["Regular Hours"].to_numpy()[:nrows]
    attr3 = df["Regular Gross Paid"].to_numpy()[:nrows]
    attr4 = df["OT Hours"].to_numpy()[:nrows]
    attr5 = df["Total OT Paid"].to_numpy()[:nrows]
    attr6 = df["Total Other Pay"].to_numpy()[:nrows]
    attr7 = df["Working Years"].to_numpy()[:nrows]

    # Stack the attributes into a 2D array
    data = np.column_stack((attr1, attr2, attr3, attr4, attr5, attr6, attr7))

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 8)])
    print(df.shape)
    return df


if __name__ == "__main__":
    generate_dataset()
