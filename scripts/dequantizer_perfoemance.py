import time

import numpy as np
from tqdm import tqdm

from colse.cat_transform import DeQuantize
from colse.cdf_storage import CDFStorage
from colse.custom_data_generator import CustomDataGen
from colse.dataset_names import DatasetNames
from colse.emphirical_cdf import EMPMethod
from colse.optimized_emp_cdf import OptimizedEmpiricalCDFModel
from colse.spline_dequantizer import SplineDequantizer

dataset_type = DatasetNames.POWER_DATA
dataset = CustomDataGen(
    no_of_rows=None,
    no_of_queries=None,
    dataset_type=dataset_type,
    data_split="train",
    selected_cols=None,
    scalar_type="min_max",
    dequantize=False,
    seed=1,
    is_range_queries=True,
    verbose=False,
)

df = dataset.df
CDF_STORAGE_CACHE = f"{dataset_type}_cdf_dataframe"

def get_query(q1, q2):
    return np.array([[q11, q22] for q11, q22 in zip(q1, q2)]).reshape(-1)


X = np.array([get_query(ql, qr) for ql, qr in zip(dataset.query_l, dataset.query_r)])
COLUMN_INDEXES = [i for i in range(dataset_type.get_no_of_columns())]

def check_spline_dequantizer_performance():
    cdf_df = SplineDequantizer()
    cdf_df.fit(df, columns=dataset_type.get_non_continuous_columns())

    start_time = time.perf_counter()
    loop = tqdm(X, total=X.shape[0])
    for query in loop:
        query = query.reshape(1, -1)
        cdf_df.get_converted_cdf(query, COLUMN_INDEXES)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}")


def check_colse_dequantizer_performance():
    cdf_df = CDFStorage(
        OptimizedEmpiricalCDFModel,
        cached_name_string=CDF_STORAGE_CACHE,
        override=False,
        emp_method=EMPMethod.RELATIVE,
        enable_low_precision=False,
        max_unique_values="auto",
    )
    cdf_df.fit(df)

    start_time = time.perf_counter()
    loop = tqdm(X, total=X.shape[0])
    for query in loop:
        query = query.reshape(1, -1)
        cdf_df.get_converted_cdf(query, COLUMN_INDEXES, nproc=1, cache=False)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    check_spline_dequantizer_performance()
    # check_colse_dequantizer_performance()
