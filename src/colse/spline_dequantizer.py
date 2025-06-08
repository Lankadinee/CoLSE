from pathlib import Path
import pickle
import time
from typing import Tuple
from dataclasses import dataclass, field
from loguru import logger
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import PchipInterpolator
from rich.console import Console
from rich.table import Table
from enum import StrEnum, auto


class DequantizerType(StrEnum):
    CATEGORICAL = auto()
    CONTINUOUS = auto()

@dataclass
class Metadata:
    df_cols: list[str] = field(default_factory=list)
    df_max_values: dict[str, float] = field(default_factory=dict)
    df_min_values: dict[str, float] = field(default_factory=dict)


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

    def __init__(self, M: int = 10000, path: Path = None):
        """
        Parameters
        ----------
        M : int
            Number of evenly spaced points in [min_value, max_value] to build the "CDF → z" lookup table
            (grid size). Larger M yields a more accurate inversion but is slightly slower.
        """
        self.M = M
        self.dequantizers = {
            DequantizerType.CATEGORICAL: {},
            DequantizerType.CONTINUOUS: {},
        }  # will hold per-column parameters
        self.metadata = Metadata()
        self.already_loaded = False
        if path and path.exists():
            self.load_from_pickle(path)
            self.already_loaded = True
        
        self.path = path

        self.time_taken_for_fit = 0


    def _fit_continuous_column(self, x: pd.Series, col_name: str):
        """
        Fit a dequantizer for a continuous column.
        """

        B = 5000
        counts, edges = np.histogram(x, bins=B, density=False)
        N = len(x)
        p = counts / float(N)            # probability in each bin
        cdf_bin = np.concatenate(([0.0], np.cumsum(p)))  # length B+1
        # edges is length B+1, e.g. edges = [x0, x1, …, xB]
        
        xs = edges.astype(np.float64)      # [x0, x1, …, xB]
        ys = cdf_bin                       # [0, cumsum(p)…, 1.0]
        pchip_cdf = PchipInterpolator(xs, ys, extrapolate=False)

        self.dequantizers[DequantizerType.CONTINUOUS][col_name] = {
            "spline_cdf": pchip_cdf,
            # "edges": edges,
            # "cdf_bin": cdf_bin,
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
        self.dequantizers[DequantizerType.CATEGORICAL][col_name] = {
            # "uniques": uniques,
            # "K": K,
            "mapping": mapping,
            "cdf_vals": cdf_vals,
            # "z_b": z_b,
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
        if self.already_loaded:
            logger.warning("Dequantizer already fitted. Skipping fit().")
            return False

        start_time = time.perf_counter()
        self.metadata.df_cols = df.columns.tolist()
        self.metadata.df_max_values = df.max().to_dict()
        self.metadata.df_min_values = df.min().to_dict()
        if columns is None:
            columns = df.columns.tolist()

        # Fit categorical columns
        for col in columns:
            logger.info(f"Fitting dequantizer for categorical column: {col}")
            self._fit_single_column(df[col], col)
            
        # Fit continuous columns
        for col in [ c for c in df.columns if c not in columns]:
            logger.info(f"Fitting dequantizer for continuous/discrete column: {col}")
            self._fit_continuous_column(df[col], col)
        
        logger.info("Dequantizer fitted successfully.")
        self.time_taken_for_fit = time.perf_counter() - start_time
        logger.info(f"Time taken for fit: {self.time_taken_for_fit:.2f} seconds")

        if self.path and not self.already_loaded:
            self.save_to_pickle(self.path)
        return True

    def save_to_pickle(self, path: str):
        save_list = [self.metadata, self.dequantizers]
        with open(path, "wb") as f:
            pickle.dump(save_list, f)
        logger.info(f"Dequantizer saved to {path}")
    

    def load_from_pickle(self, path: str):
        with open(path, "rb") as f:
            self.metadata, self.dequantizers = pickle.load(f)
        logger.info(f"Dequantizer loaded from {path}")
        
    def _transform_single_column(self, x: pd.Series, col_name: str) -> np.ndarray:
        """
        Dequantize one column into a continuous NumPy array (dtype=float64).
        Returns an array of shape (N,) with values on the original scale of z_b.
        """
        params = self.dequantizers[DequantizerType.CATEGORICAL][col_name]
        mapping = params["mapping"]
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
        Currently only supports dequantizing Categorical columns.
        TODO: Add support for Discrete columns.?!

        Dequantize each specified column in df and return a new DataFrame
        containing only the continuous (dequantized) versions of those columns.
        """
        start_time = time.perf_counter()
        if columns is None:
            columns = list(self.dequantizers[DequantizerType.CATEGORICAL].keys())
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            if col in columns:
                result[col] = self._transform_single_column(df[col], col)
            else:
                result[col] = df[col]
        time_taken_for_transform = time.perf_counter() - start_time

        table = Table(title="Time taken for Dequantizer")
        table.add_column("Type", justify="right")
        table.add_column("Time taken (in seconds)", justify="right")
        table.add_row("Transform", f"{time_taken_for_transform:.2f}")
        table.add_row("Fit", f"{self.time_taken_for_fit:.2f}")
        table.add_row("Fit + Transform", f"{time_taken_for_transform + self.time_taken_for_fit:.2f}")
        console = Console()
        console.print(table)
        return result
    
    def get_cdf_values(self, col_name: str, original_value) -> float:
        """
        Given one old-data value, return the cumulative frequency at that value.
        """
        if float(original_value) >= self.metadata.df_max_values[col_name]:
            return 1
        
        if float(original_value) <= self.metadata.df_min_values[col_name]:
            return 0
        
        if col_name in self.dequantizers[DequantizerType.CONTINUOUS]:
            return self.dequantizers[DequantizerType.CONTINUOUS][col_name]["spline_cdf"](float(original_value))
            
        spline = self.dequantizers[DequantizerType.CATEGORICAL][col_name]["spline_cdf"]
        cdf_at_v = float(spline(original_value))
        return cdf_at_v
    
    def get_cdf_values_for_cat(self, col_name: str, original_value: str, ub=True) -> float:
        """
        Given a query and a column index, return the cumulative frequency at that value.
        """
        if original_value in [-np.inf, np.str_("-inf")]:
            return 0
        if original_value in [np.inf, np.str_("inf")]:
            return 1
        code = self.dequantizers[DequantizerType.CATEGORICAL][col_name]["mapping"][original_value]
        if ub:
            return self.dequantizers[DequantizerType.CATEGORICAL][col_name]["cdf_vals"][code]
        else:
            if code == 0:
                return 0
            return self.dequantizers[DequantizerType.CATEGORICAL][col_name]["cdf_vals"][code - 1]
        
    
    def get_converted_cdf(self, query, column_indexes):
        """Convert a query into continuous CDF values."""

        cdf_values = []
        categorical_columns = self.dequantizers[DequantizerType.CATEGORICAL].keys()
        for idx, value in enumerate(query[0]):
            col_name = self.metadata.df_cols[column_indexes[idx // 2]]

            if col_name in categorical_columns:
                ub = True if idx % 2 == 1 else False
                cdf_values.append(self.get_cdf_values_for_cat(col_name, value, ub=ub))
            else:
                cdf_values.append(self.get_cdf_values(col_name, value))

        return np.array(cdf_values)
    
    def get_mapped_query(self, query, column_indexes):
        """Convert categorical columns in a query into their mapped values"""
        mapped_query = []
        categorical_columns = self.dequantizers[DequantizerType.CATEGORICAL].keys()
        for idx, value in enumerate(query[0]):
            col_name = self.metadata.df_cols[column_indexes[idx // 2]]
            if col_name in categorical_columns:
                if value == np.str_("-inf"):
                    mapped_query.append(min(self.dequantizers[DequantizerType.CATEGORICAL][col_name]["mapping"].values()))
                elif value == np.str_("inf"):
                    mapped_query.append(max(self.dequantizers[DequantizerType.CATEGORICAL][col_name]["mapping"].values()))
                else:
                    mapped_query.append(self.dequantizers[DequantizerType.CATEGORICAL][col_name]["mapping"][value])
            else:
                mapped_query.append(np.float64(value))
        return np.array(mapped_query)