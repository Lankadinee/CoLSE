



import numpy as np
import pandas as pd

from colse.spline_dequantizer import SplineDequantizer


def create_data():
    df = pd.DataFrame({
        'x': np.arange(1000),
        'y': np.arange(1000),
    })
    return df


def test_spline():
    df = create_data()
    dequantizer = SplineDequantizer()
    dequantizer.fit(df)
    dequantized_df = dequantizer.transform(df)
    print(dequantized_df.head())

if __name__ == "__main__":
    test_spline()
