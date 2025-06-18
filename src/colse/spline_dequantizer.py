from datetime import datetime
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import StrEnum, auto

import numpy as np
import pandas as pd
from loguru import logger
from pandas.api.types import is_numeric_dtype
from rich.console import Console
from rich.table import Table
from scipy.interpolate import PchipInterpolator

from colse.data_path import DataPathDir, get_data_path
from colse.dataset_names import DatasetNames
from colse.df_utils import save_dataframe


# set logger level to DEBUG
logger.level("INFO")

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

    def __init__(
        self, dataset_type: DatasetNames, 
        m: int = 10000, 
        cache_name: str = None,
        output_file_name: str = None
    ):
        """
        Parameters
        ----------
        M : int
            Number of evenly spaced points in [min_value, max_value] to build the "CDF → z" lookup table
            (grid size). Larger M yields a more accurate inversion but is slightly slower.
        """
        self._m = m
        self._dequantizers = {
            DequantizerType.CATEGORICAL: {},
            DequantizerType.CONTINUOUS: {}
        }  # will hold per-column parameters
        self._metadata = Metadata()
        self._dataset_type = dataset_type
        if cache_name:
            path = get_data_path(DataPathDir.CDF_CACHE, dataset_type.value) / f"{cache_name}"
        else:
            path = None

        self._already_loaded = False
        if path and path.exists():
            self.load_from_pickle(path)
            self._already_loaded = True
            logger.warning(f"Dequantizer cache found and loaded from:{path}")

        self._cache_path = path

        self._time_taken_for_fit = 0
        self._out_file_name = output_file_name if output_file_name else "dequantized_v2.parquet"
        self._dequantized_dataset_path = (
            get_data_path(dataset_type.value) / self._out_file_name
        )

    def get_dequantized_dataset_name(self):
        if self._dequantized_dataset_path.exists():
            return self._out_file_name
        else:
            return None

    def _fit_continuous_column(self, x: pd.Series, col_name: str):
        """
        Fit a dequantizer for a continuous column.
        """

        B = 5000
        counts, edges = np.histogram(x, bins=B, density=False)
        N = len(x)
        p = counts / float(N)  # probability in each bin
        cdf_bin = np.concatenate(([0.0], np.cumsum(p)))  # length B+1
        # edges is length B+1, e.g. edges = [x0, x1, …, xB]

        xs = edges.astype(np.float64)  # [x0, x1, …, xB]
        ys = cdf_bin  # [0, cumsum(p)…, 1.0]
        pchip_cdf = PchipInterpolator(xs, ys, extrapolate=False)

        self._dequantizers[DequantizerType.CONTINUOUS][col_name] = {
            "is_one_value": False,
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
        logger.info(f"X shape: {x.shape}")
        if is_numeric_dtype(x):
            logger.info("Numeric column")
            uniques = np.sort(x.dropna().unique())
            K = len(uniques)
            # logger.info(f"Uniques shape: {uniques.shape}")
            # logger.info(f"K: {K}")
            mapping = {val: idx for idx, val in enumerate(uniques)}
            codes = x.map(mapping).values
            N = len(codes)
            counts = np.bincount(codes, minlength=K)
            p = counts.astype(np.float64) / float(N)
            cdf_vals = np.cumsum(p)
            z_b = uniques.astype(np.float64)
        else:
            logger.info("Categorical column")
            codes, uniques = pd.factorize(x, sort=True)
            if (codes < 0).any():
                raise ValueError(
                    f"Column '{col_name}' contains NaN or unseen categories during fit."
                )
            K = len(uniques)
            # logger.info(f"Uniques shape: {uniques.shape}")
            # logger.info(f"K: {K}")
            mapping = {val: idx for idx, val in enumerate(uniques)}
            N = len(codes)
            counts = np.bincount(codes, minlength=K)
            p = counts.astype(np.float64) / float(N)
            cdf_vals = np.cumsum(p)
            z_b = np.arange(K, dtype=np.float64)

        # logger.info(f"CDF values shape: {cdf_vals.shape}")
        # logger.info(f"Z_b values shape: {z_b.shape}")
        # logger.info(f"ConditionK {K}: {K == 1}")
        if K == 1:
            self._dequantizers[DequantizerType.CATEGORICAL][col_name] = {
                "mapping": mapping,
                "cdf_vals": cdf_vals,
                "is_one_value": True,
            }
        else:
            spline_cdf = PchipInterpolator(z_b, cdf_vals, extrapolate=False)
            grid_z = np.linspace(z_b[0], z_b[-1], self._m, dtype=np.float64)
            grid_c = spline_cdf(grid_z)
            self._dequantizers[DequantizerType.CATEGORICAL][col_name] = {
                "is_one_value": False,
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
        if self._already_loaded:
            logger.warning("Dequantizer already fitted. Skipping fit().")
            return False

        start_time = time.perf_counter()
        self._metadata.df_cols = df.columns.tolist()
        self._metadata.df_max_values = df.max().to_dict()
        self._metadata.df_min_values = df.min().to_dict()
        if columns is None:
            columns = df.columns.tolist()

        # Fit categorical columns
        for col in columns:
            logger.info(f"Fitting dequantizer for categorical column: {col}")
            self._fit_single_column(df[col], col)

        # Fit continuous columns
        for col in [c for c in df.columns if c not in columns]:
            logger.info(f"Fitting dequantizer for continuous column: {col}")
            self._fit_continuous_column(df[col], col)

        logger.info("Dequantizer fitted successfully.")
        self._time_taken_for_fit = time.perf_counter() - start_time
        logger.info(f"Time taken for fit: {self._time_taken_for_fit:.2f} seconds")

        if self._cache_path and not self._already_loaded:
            self.save_to_pickle(self._cache_path)
        return True

    def save_to_pickle(self, path: str):
        save_list = [self._metadata, self._dequantizers]
        with open(path, "wb") as f:
            pickle.dump(save_list, f)
        logger.info(f"Dequantizer saved to {path}")

    def load_from_pickle(self, path: str):
        with open(path, "rb") as f:
            self._metadata, self._dequantizers = pickle.load(f)
        logger.info(f"Dequantizer loaded from {path}")

    def _transform_single_column(self, x: pd.Series, col_name: str) -> np.ndarray:
        """
        Dequantize one column into a continuous NumPy array (dtype=float64).
        Returns an array of shape (N,) with values on the original scale of z_b.
        """
        params = self._dequantizers[DequantizerType.CATEGORICAL][col_name]
        if params["is_one_value"]:
            mapping = params["mapping"]
            # logger.debug(f"Mapping: {mapping}")
            values = x.map(mapping)
            # logger.debug(f"Values: {values}, unique: {values.unique()}")
            return values
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Currently only supports dequantizing Categorical columns.

        Dequantize each specified column in df and return a new DataFrame
        containing only the continuous (dequantized) versions of those columns.
        """
        start_time = time.perf_counter()
        
        cat_cols = self._dataset_type.get_non_continuous_columns() 
        result = pd.DataFrame(index=df.index)
        table_v1 = Table(title="Time taken for Dequantizer")
        table_v1.add_column("Column Name", justify="right")
        table_v1.add_column("Time taken (in seconds)", justify="right")
        for col in df.columns:
            start_time_transform = time.perf_counter()
            if col in cat_cols:
                result[col] = self._transform_single_column(df[col], col)
            else:
                result[col] = df[col]
            time_taken_for_transform = time.perf_counter() - start_time_transform
            table_v1.add_row(col, f"{time_taken_for_transform:.2f}")
        time_taken_for_total_transform = time.perf_counter() - start_time

        table = Table(title="Total Time taken for Dequantizer")
        table.add_column("Type", justify="right")
        table.add_column("Time taken (in seconds)", justify="right")
        table.add_row("Transform", f"{time_taken_for_total_transform:.2f}")
        table.add_row("Fit", f"{self._time_taken_for_fit:.2f}")
        table.add_row(
            "Fit + Transform",
            f"{time_taken_for_total_transform + self._time_taken_for_fit:.2f}",
        )
        console = Console()
        console.print(table_v1)
        console.print(table)
        # Save dequantized dataset
        save_dataframe(result, self._dequantized_dataset_path)
        logger.info(f"Saved dequantized dataset to {self._dequantized_dataset_path}")
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        non_continuous_columns = self._dataset_type.get_non_continuous_columns()
        self.fit(df, columns=non_continuous_columns)
        if len(non_continuous_columns) > 0:
            logger.info(f"Dequantizing {len(non_continuous_columns)} non-continuous columns")
            return self.transform(df)
        else:
            logger.info("No non-continuous columns to dequantize")
            return df


    def  _get_cdf_values(self, col_name: str, original_value) -> float:
        """
        Given one old-data value, return the cumulative frequency at that value.
        """
        if float(original_value) >= self._metadata.df_max_values[col_name]:
            return 1

        if float(original_value) <= self._metadata.df_min_values[col_name]:
            return 0

        _metadata = self._dequantizers[DequantizerType.CONTINUOUS][col_name]
        return _metadata["spline_cdf"](float(original_value))


    def _get_cdf_values_for_cat(
        self, col_name: str, original_value: str, ub=True
    ) -> float:
        """
        Given a query and a column index, return the cumulative frequency at that value.
        """
        
        if original_value in [-np.inf, np.str_("-inf")]:
            return 0
        if original_value in [np.inf, np.str_("inf")]:
            return 1
        
        _metadata = self._dequantizers[DequantizerType.CATEGORICAL][col_name]
        try:
            mapping = _metadata["mapping"]
            code = mapping[original_value] if original_value in mapping else mapping[str(original_value)]
        except KeyError:
            logger.warning(f"Value {original_value} not found in {col_name} mapping for column {mapping.keys()}")
            return 0
            # raise ValueError(f"Value {original_value} not found in mapping for column {mapping.keys()}")
        
        if ub:
            return _metadata["cdf_vals"][code]
        else:
            if code == 0:
                return 0
            return _metadata["cdf_vals"][code - 1]


    def _get_cdf_values_for_descrete(self, col_name: str, original_value: str, strict=False) -> float:
        """
        Given a query and a column index, return the cumulative frequency at that value.
        """
        meta = self._dequantizers[DequantizerType.DISCRETE][col_name]
        # find the index: the largest `i` such that uniques[i] == original_value
        idx = np.searchsorted(meta["uniques"], float(original_value))
        if idx >= len(meta["uniques"]):
            # unseen value: handle as 0 or 1 or via spline fallback
            return 1
        elif idx == 0:
            return 0
        # now return strict vs non-strict CDF
        return meta["cdf_values_strict"][idx] if strict else meta["cdf_vals"][idx]
    

    def get_converted_cdf(self, query, column_indexes):
        """Convert a query into continuous CDF values."""

        cdf_values = []
        categorical_columns = self._dataset_type.get_non_continuous_columns()
        pairwise_query = query.reshape(-1, 2)
        for idx, (value_lb, value_ub) in enumerate(pairwise_query):
            col_name = self._metadata.df_cols[column_indexes[idx]]
            if col_name in categorical_columns:
                cdf_values.append(self._get_cdf_values_for_cat(col_name, value_lb, ub=False))
                cdf_values.append(self._get_cdf_values_for_cat(col_name, value_ub, ub=True))
            else:
                cdf_values.append(self._get_cdf_values(col_name, value_lb))
                cdf_values.append(self._get_cdf_values(col_name, value_ub))

        return np.clip(np.array(cdf_values), 0, 1)

    # this is public method
    def get_mapped_query(self, query, column_indexes):
        """
        Convert the query into a mapped query.
        Use each column's mapping to convert the query into a mapped query.
        """
        mapped_query = []
        _metadata = None
        categorical_columns = self._dataset_type.get_non_continuous_columns()
        try:
            for idx, value in enumerate(query[0]):
                col_name = self._metadata.df_cols[column_indexes[idx//2]]
                if col_name in categorical_columns:
                    _metadata = self._dequantizers[DequantizerType.CATEGORICAL][col_name]
                    if value == np.str_("-inf"):
                        mapped_query.append(
                            min(
                                _metadata["mapping"].values()
                            )
                        )
                    elif value == np.str_("inf"):
                        mapped_query.append(
                            max(
                                _metadata["mapping"].values()
                            )
                        )
                    else:
                        mapped_query.append(
                            _metadata["mapping"][value]
                        )
                else:
                    mapped_query.append(np.float64(value))
        except KeyError:
            keys = _metadata["mapping"].keys() if _metadata else None
            logger.info(f"Query: {query} Column Indexes: {column_indexes} Column Name: {col_name} All columns: {self._metadata.df_cols}")
            logger.exception(f"Value {value} not found in {col_name} mapping for column {keys}")
            raise ValueError(f"Value {value} not found in {col_name} mapping for column {keys}")
        
        return np.array(mapped_query)