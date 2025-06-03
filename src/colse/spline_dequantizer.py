from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from scipy.interpolate import PchipInterpolator


class SplineDequantizer:
    """
    Implements spline-based dequantization (via PCHIP) on discrete/categorical columns
    of a Pandas DataFrame, with no iterative model training—just histogram → spline fit →
    lookup → vectorized inversion.

    New methods added:
      • get_continuous_interval(column, original_value)
           → returns the continuous [low, high) interval in [0,1) corresponding to a single
             old-value (e.g. a category or discrete integer).
      • get_continuous_intervals(column, original_values)
           → returns a list of [low, high) intervals for each value in the iterable original_values.
    """

    def __init__(self, M: int = 10000):
        """
        Parameters
        ----------
        M : int
            Number of evenly spaced points in [min_value, max_value] to build the "CDF → z" lookup table
            (grid size). Larger M yields a more accurate inversion but is slightly slower.
        """
        self.M = M
        self.dequantizers = {}  # will hold per-column parameters

    def _fit_single_column(self, x: pd.Series, col_name: str):
        """
        Build histogram, CDF, PCHIP spline, and lookup table for one column.
        Saves in self.dequantizers[col_name]:
            - uniques : array of distinct values (sorted)
            - K       : number of distinct levels
            - mapping : dict mapping original values → indices 0..K-1
            - cdf_vals: array of length K giving cumulative frequency at each unique value
            - z_b     : array of length K holding the unique values (for spline knots)
            - grid_z  : M linearly spaced points between min and max unique values
            - grid_c  : spline_cdf(grid_z), used for fast inversion
        """
        if is_integer_dtype(x):
            uniques = np.sort(x.dropna().unique())
            K = len(uniques)
            mapping = {val: idx for idx, val in enumerate(uniques)}
            codes = x.map(mapping).values
            N = len(codes)
            # Histogram on codes
            counts = np.bincount(codes, minlength=K)
            p = counts.astype(np.float64) / float(N)
            cdf_vals = np.cumsum(p)  # length K
            # Use unique integer values as z_b positions
            z_b = uniques.astype(np.float64)  # shape (K,)
        else:
            codes, uniques = pd.factorize(x, sort=True)
            if (codes < 0).any():
                raise ValueError(
                    f"Column '{col_name}' contains NaN or unseen categories during fit."
                )
            K = len(uniques)
            mapping = {val: idx for idx, val in enumerate(uniques)}
            N = len(codes)
            counts = np.bincount(codes, minlength=K)
            p = counts.astype(np.float64) / float(N)
            cdf_vals = np.cumsum(p)
            # Use integer indices as z_b for categorical values
            z_b = np.arange(K, dtype=np.float64)

        # Build CDF boundary array (for interval lookup)
        # For simplicity, we treat cdf_vals[i] as CDF at z_b[i]
        # We'll use these for sampling and spline knot points.

        # Fit PCHIP spline for CDF(z)
        spline_cdf = PchipInterpolator(z_b, cdf_vals, extrapolate=False)
        # Build grid on value scale
        grid_z = np.linspace(z_b[0], z_b[-1], self.M, dtype=np.float64)
        grid_c = spline_cdf(grid_z)
        # Store parameters
        self.dequantizers[col_name] = {
            "uniques": uniques,
            "K": K,
            "mapping": mapping,
            "cdf_vals": cdf_vals,
            "z_b": z_b,
            "grid_z": grid_z,
            "grid_c": grid_c,
        }

    def fit(self, df: pd.DataFrame, columns=None):
        """
        Fit dequantizers for each specified column.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list[str] or None
            If None, fit on all columns present in df. Otherwise, fit only on df[columns].
        """
        if columns is None:
            columns = df.columns.tolist()
        for col in columns:
            self._fit_single_column(df[col], col)

    def _transform_single_column(self, x: pd.Series, col_name: str) -> np.ndarray:
        """
        Dequantize one column into a continuous NumPy array (dtype=float64).
        Returns an array of shape (N,) with values on the original scale of z_b.
        """
        params = self.dequantizers[col_name]
        uniques = params["uniques"]
        K = params["K"]
        mapping = params["mapping"]
        cdf_vals = params["cdf_vals"]
        z_b = params["z_b"]
        grid_z = params["grid_z"]
        grid_c = params["grid_c"]
        # Map to index codes
        try:
            codes = x.map(mapping).values
        except Exception:
            unseen = x[~x.isin(uniques)].unique().tolist()
            raise ValueError(
                f"Column '{col_name}' contains values not seen during fit: {unseen}"
            )
        # For each code i, CDF interval is [cdf_vals[i-1] (or 0), cdf_vals[i]]
        lows = np.concatenate(([0.0], cdf_vals[:-1]))[codes]
        highs = cdf_vals[codes]
        # Draw uniform random v in [lows, highs]
        v = np.random.uniform(lows, highs)
        # Invert CDF: z = interp(v, grid_c, grid_z)
        z = np.interp(v, grid_c, grid_z)
        return z

    def transform(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Dequantize each specified column in df and return a new DataFrame
        containing only the continuous (dequantized) versions of those columns.
        """
        if columns is None:
            columns = list(self.dequantizers.keys())
        result = pd.DataFrame(index=df.index)
        for col in columns:
            result[col] = self._transform_single_column(df[col], col)
        return result

    def get_continuous_interval(
        self, col_name: str, original_value
    ) -> Tuple[float, float]:
        """
        Given one old-data value, return the continuous interval [low, high)
        on the original value scale corresponding to that value.
        """
        if col_name not in self.dequantizers:
            raise KeyError(f"Column '{col_name}' was not fit. Call fit(...) first.")
        params = self.dequantizers[col_name]
        uniques = params["uniques"]
        K = params["K"]
        mapping = params["mapping"]
        cdf_vals = params["cdf_vals"]
        z_b = params["z_b"]
        if original_value not in mapping:
            raise ValueError(
                f"Value {original_value!r} not found in column '{col_name}' (fit saw {len(uniques)} uniques)."
            )
        idx = mapping[original_value]
        low_cdf = cdf_vals[idx - 1] if idx > 0 else 0.0
        high_cdf = cdf_vals[idx]
        # Corresponding z interval is [z_b[idx-1], z_b[idx]] (or single point if idx=0)
        low_z = z_b[idx - 1] if idx > 0 else z_b[0]
        high_z = z_b[idx]
        return (low_z, high_z)
