from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from colse.cdf_storage import CDFStorage
from colse.custom_data_generator import CustomDataGen
from colse.dataset_names import DatasetNames
from colse.df_utils import load_dataframe
from colse.emphirical_cdf import EMPMethod
from colse.optimized_emp_cdf import OptimizedEmpiricalCDFModel
from colse.spline_dequantizer import SplineDequantizer

dataset_type = DatasetNames.DMV_DATA
NO_OF_ROWS = None

dataset = CustomDataGen(
    no_of_rows=NO_OF_ROWS,
    no_of_queries=None,
    data_file_name=None,
    dataset_type=dataset_type,
    data_split="train",
    selected_cols=None,
    scalar_type="min_max",  # 'min_max' or 'standard
    dequantize=False,
    enable_query_dequantize=False,
    seed=1,
    is_range_queries=True,
    verbose=False,
)

COLUMN_INDEXES = [i for i in range(dataset_type.get_no_of_columns())]
no_of_rows = dataset.no_of_rows
query_l = dataset.query_l
query_r = dataset.query_r
actual_ce_ds = dataset.true_card

# loop = tqdm(enumerate(zip(query_l, query_r, actual_ce_ds)), total=query_l.shape[0])


def get_query(q1, q2):
    return np.array([[q11, q22] for q11, q22 in zip(q1, q2)]).reshape(-1)


X = np.array([get_query(ql, qr) for ql, qr in zip(query_l, query_r)])
y = np.array(actual_ce_ds) / no_of_rows
loop = tqdm(zip(X, y), total=X.shape[0])


@dataclass
class ActualCDF:
    df: pd.DataFrame
    df_cols: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.df_cols = list(self.df.columns)

    def get_cdf_values(self, col_name, original_value):
        if isinstance(original_value, str):
            return self.df[self.df[col_name] == original_value].shape[0] / self.df.shape[0]
        else:
            return self.df[self.df[col_name] <= original_value].shape[0] / self.df.shape[0]

    def get_converted_cdf(self, query, column_indexes):
        cdf_list = np.array(
            [
                self.get_cdf_values(
                    self.df_cols[column_indexes[idx // 2]], query[0][idx]
                )
                for idx in range(query.shape[1])
            ]
        )
        return cdf_list


def main():
    # Dequantize dataset
    s_dequantize = SplineDequantizer()
    dataset_path = dataset_type.get_file_path()
    df = load_dataframe(dataset_path)
    col_names = list(df.columns)
    s_dequantize.fit(df, columns=dataset_type.get_non_continuous_columns())

    cdf_df = CDFStorage(
        OptimizedEmpiricalCDFModel,
        cached_name_string=None,
        override=True,
        emp_method=EMPMethod.RELATIVE,
        enable_low_precision=False,
        max_unique_values="auto",
    )
    cdf_df.fit(df)

    actual_cdf = ActualCDF(df)

    diff_list_v2 = []
    diff_list_v1 = []
    max_iterations = 10
    iteration = 0
    for idx, (query, y_act) in enumerate(loop):
        query = query.reshape(1, -1)
        cdf_list_v1 = cdf_df.get_converted_cdf(
            query, COLUMN_INDEXES, nproc=1, cache=False
        )
        cdf_list_v2 = s_dequantize.get_converted_cdf(query, COLUMN_INDEXES)
        cdf_list_v3 = actual_cdf.get_converted_cdf(query, COLUMN_INDEXES)
        diff_list_v1.append(np.abs(cdf_list_v1 - cdf_list_v3))
        diff_list_v2.append(np.abs(cdf_list_v2 - cdf_list_v3))
        iteration += 1
        if iteration >= max_iterations:
            break

    diff_list_v1 = np.array(diff_list_v1)
    diff_list_v2 = np.array(diff_list_v2)

    max_diff_v1 = np.max(diff_list_v1.reshape(iteration, -1, 2), axis=2)
    max_diff_v2 = np.max(diff_list_v2.reshape(iteration, -1, 2), axis=2)

    percentiles = [0, 25, 50, 75, 90, 99, 100]
    percentiles_v1 = np.percentile(max_diff_v1, percentiles, axis=0)
    percentiles_v2 = np.percentile(max_diff_v2, percentiles, axis=0)

    table = Table(title="Dequantizer Test")
    table.add_column("Percentile", justify="right")
    for i in range(len(COLUMN_INDEXES)):
        table.add_column(f"v1_{col_names[i]}", justify="right")
        table.add_column(f"v2_{col_names[i]}", justify="right")
    table.add_column(f"v1_max", justify="right")
    table.add_column(f"v2_max", justify="right")
    for i in range(len(percentiles_v1)):
        row_data = [f"{percentiles[i]}"]
        for j in range(len(percentiles_v1[i])):
            row_data.extend(
                [f"{percentiles_v1[i][j]:.4f}", f"{percentiles_v2[i][j]:.4f}"]
            )
        row_data.extend(
            [f"{np.max(percentiles_v1[i]):.4f}", f"{np.max(percentiles_v2[i]):.4f}"]
        )
        table.add_row(*row_data)
    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
