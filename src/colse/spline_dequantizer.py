from typing import Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import PchipInterpolator


class SplineDequantizer:
    """
    Implements spline-based dequantization (via PCHIP) on discrete/categorical columns
    of a Pandas DataFrame, with no iterative model training—just histogram → spline fit →
    lookup → vectorized inversion.

    New methods added:
      • get_continuous_interval(column, original_value)
           → returns the continuous [low, high) interval on the original value scale
             corresponding to a given value. If the value was unseen but integer, interpolate neighbors.
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
        self.continuous_dequantizers = {}
        self.df_cols = []
        self.df_max_values = {}
        self.df_min_values = {}
        self.categorical_columns = []

    def _fit_continuous_column(self, x: pd.Series, col_name: str):
        """
        Fit a dequantizer for a continuous column.
        """
        # self.dequantizers[col_name] = {}
        # z_b = np.sort(x.dropna().unique())  # Ensure values are sorted
        # cdf_vals = np.cumsum(z_b) / np.sum(z_b)
        # self.continuous_dequantizers[col_name] = {}
        # self.continuous_dequantizers[col_name]['spline_cdf'] = PchipInterpolator(z_b, cdf_vals, extrapolate=False)

        B = 5000
        counts, edges = np.histogram(x, bins=B, density=False)
        N = len(x)
        p = counts / float(N)            # probability in each bin
        cdf_bin = np.concatenate(([0.0], np.cumsum(p)))  # length B+1
        # edges is length B+1, e.g. edges = [x0, x1, …, xB]
        
        xs = edges.astype(np.float64)      # [x0, x1, …, xB]
        ys = cdf_bin                       # [0, cumsum(p)…, 1.0]
        pchip_cdf = PchipInterpolator(xs, ys, extrapolate=False)

        self.continuous_dequantizers[col_name] = {
            "spline_cdf": pchip_cdf,
            "edges": edges,
            "cdf_bin": cdf_bin,
        }


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
        if is_numeric_dtype(x):
            uniques = np.sort(x.dropna().unique())
            K = len(uniques)
            mapping = {val: idx for idx, val in enumerate(uniques)}
            codes = x.map(mapping).values
            N = len(codes)
            counts = np.bincount(codes, minlength=K)
            p = counts.astype(np.float64) / float(N)
            cdf_vals = np.cumsum(p)
            z_b = uniques.astype(np.float64)
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
            z_b = np.arange(K, dtype=np.float64)

        spline_cdf = PchipInterpolator(z_b, cdf_vals, extrapolate=False)
        grid_z = np.linspace(z_b[0], z_b[-1], self.M, dtype=np.float64)
        grid_c = spline_cdf(grid_z)
        self.dequantizers[col_name] = {
            "uniques": uniques,
            "K": K,
            "mapping": mapping,
            "cdf_vals": cdf_vals,
            "z_b": z_b,
            "spline_cdf": spline_cdf,
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
        self.df_cols = df.columns.tolist()
        self.df_max_values = df.max().to_dict()
        self.df_min_values = df.min().to_dict()
        if columns is None:
            columns = df.columns.tolist()
        else:
            self.categorical_columns = columns

        for col in columns:
            self._fit_single_column(df[col], col)
            
        
        for col in [ c for c in df.columns if c not in columns]:
            self._fit_continuous_column(df[col], col)
        
    def _transform_single_column(self, x: pd.Series, col_name: str) -> np.ndarray:
        """
        Dequantize one column into a continuous NumPy array (dtype=float64).
        Returns an array of shape (N,) with values on the original scale of z_b.
        """
        params = self.dequantizers[col_name]
        mapping = params["mapping"]
        z_b = params["z_b"]
        grid_z = params["grid_z"]
        grid_c = params["grid_c"]
        cdf_vals = params["cdf_vals"]

        codes = x.map(mapping)
        if codes.isna().any():
            unseen = x[codes.isna()].unique().tolist()
            raise ValueError(
                f"Column '{col_name}' contains values not seen during fit: {unseen}"
            )
        codes = codes.values.astype(int)

        lows = np.concatenate(([0.0], cdf_vals[:-1]))[codes]
        highs = cdf_vals[codes]
        v = np.random.uniform(lows, highs)
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
        for col in df.columns:
            if col in columns:
                result[col] = self._transform_single_column(df[col], col)
            else:
                result[col] = df[col]
        return result
    
    def get_cdf_values(self, col_name: str, original_value) -> float:
        """
        Given one old-data value, return the cumulative frequency at that value.
        """
        is_not_cat = col_name not in self.categorical_columns
        if original_value == -np.inf or (is_not_cat and float(original_value) >= self.df_max_values[col_name]):
            return 1
        
        if original_value == np.inf or (is_not_cat and float(original_value) <= self.df_min_values[col_name]):
            return 0
        
        if col_name in self.continuous_dequantizers:
            return self.continuous_dequantizers[col_name]['spline_cdf'](float(original_value))
            
        spline = self.dequantizers[col_name]['spline_cdf']
        cdf_at_v = float(spline(original_value))
        return cdf_at_v
    
    def get_cdf_values_for_cat(self, col_name: str, original_value: str, ub=True) -> float:
        """
        Given a query and a column index, return the cumulative frequency at that value.
        """
        if original_value in [-np.inf, np.str_('-inf')]:
            return 0
        if original_value in [np.inf, np.str_('inf')]:
            return 1
        code = self.dequantizers[col_name]['mapping'][original_value]
        if ub:
            return self.dequantizers[col_name]['cdf_vals'][code]
        else:
            if code == 0:
                return 0
            return self.dequantizers[col_name]['cdf_vals'][code - 1]
        

    def get_continuous_interval(self, col_name: str, original_value) -> Tuple[float, float]:
        """
        Given one old-data value, return the continuous interval [low, high)
        on the original value scale corresponding to that value. If unseen integer,
        interpolate neighbors; otherwise require seen.
        """
        if col_name not in self.dequantizers:
            raise KeyError(f"Column '{col_name}' was not fit. Call fit(...) first.")
        params = self.dequantizers[col_name]
        mapping = params["mapping"]
        z_b = params["z_b"]

        if original_value in mapping:
            idx = mapping[original_value]
            low_z = z_b[idx]
            high_z = z_b[idx + 1] if idx + 1 < len(z_b) else z_b[idx]
            return (low_z, high_z)
        # Unseen: must be integer for interpolation
        try:
            val = int(original_value)
        except:
            raise ValueError(
                f"Value {original_value!r} not valid for column '{col_name}'."
            )
        if not is_numeric_dtype(type(val)):
            raise ValueError(
                f"Unseen non-integer value {original_value!r} for column '{col_name}'."
            )
        if val <= z_b[0]:
            return (z_b[0], z_b[1] if len(z_b) > 1 else z_b[0])
        if val >= z_b[-1]:
            return (z_b[-2] if len(z_b) > 1 else z_b[-1], z_b[-1])
        right_idx = np.searchsorted(z_b, val)
        left_idx = right_idx - 1
        low_z = z_b[left_idx]
        high_z = z_b[right_idx]
        return (low_z, high_z)
    
    def get_converted_cdf(self, query, column_indexes):
        """Convert a query into continuous CDF values."""

        cdf_values = []
        for idx, value in enumerate(query[0]):
            col_name = self.df_cols[column_indexes[idx // 2]]

            if col_name in self.categorical_columns:
                ub = True if idx % 2 == 1 else False
                cdf_values.append(self.get_cdf_values_for_cat(col_name, value, ub=ub))
            else:
                cdf_values.append(self.get_cdf_values(col_name, value))

        return np.array(cdf_values)

        # return np.array([
        #     self.get_cdf_values(self.df_cols[column_indexes[idx // 2]], query[0][idx])
        #     for idx in range(query.shape[1])
        # ])
