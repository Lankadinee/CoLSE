import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from colse.cdf_storage import CDFStorage
from colse.copula_types import CopulaTypes
from colse.custom_data_generator import CustomDataGen
from colse.data_path import get_data_path, get_excel_path, get_log_path, get_model_path
from colse.dataset_names import DatasetNames
from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive
from colse.emphirical_cdf import EMPMethod
from colse.optimized_emp_cdf import OptimizedEmpiricalCDFModel
from colse.q_error import qerror
from colse.theta_storage import ThetaStorage
from error_comp_network import ErrorCompensationNetwork

current_dir = Path(__file__).resolve().parent
iso_time_str = datetime.now().isoformat()
iso_time_str = iso_time_str.replace(":", "-")
logs_dir = get_log_path()
logger.add(
    logs_dir.joinpath(f"training-{iso_time_str}.log"),
    rotation="1 MB",
    level="DEBUG",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Divine Copula Dynamic Recursive Test"
    )
    parser.add_argument(
        "--data_split", type=str, default="train", help="Path to the testing Excel file"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="forest", help="Name of the dataset"
    )
    parser.add_argument(
        "--max_unique_values",
        type=str,
        default="auto",
        help="Size of the unique values",
    )
    return parser.parse_args()


def main():
    parsed_args = parse_args()
    IS_ERROR_COMP_TRAIN = parsed_args.data_split == "train"
    logger.info(f"IS_ERROR_COMP_TRAIN: {IS_ERROR_COMP_TRAIN}")

    max_unique_values = (
        int(parsed_args.max_unique_values)
        if parsed_args.max_unique_values != "auto"
        else "auto"
    )
    data_split = parsed_args.data_split
    dataset_type = DatasetNames(parsed_args.dataset_name)
    NO_OF_ROWS = None
    QUERY_SIZE = None
    COLUMN_INDEXES = [i for i in range(dataset_type.get_no_of_columns())]
    NO_OF_COLUMNS = len(COLUMN_INDEXES)

    COPULA_TYPE = CopulaTypes.GUMBEL
    CDF_STORAGE_CACHE = f"{dataset_type}_cdf_dataframe"
    THETA_STORAGE_CACHE = (
        get_data_path("theta_cache")
        / f"{dataset_type}_{COPULA_TYPE}_{NO_OF_COLUMNS}.pkl"
    )
    THETA_STORAGE_CACHE = None
    EXCEL_FILE_PATH = (
        get_excel_path() / f"dvine_v1_{dataset_type.value}_{data_split}_sample.xlsx"
    )
    CDF_STORAGE_CACHE_OVERRIDE = False

    dataset = CustomDataGen(
        no_of_rows=NO_OF_ROWS,
        no_of_queries=None,
        data_file_name="dmv_dequantized.parquet",
        dataset_type=dataset_type,
        data_split=data_split,
        selected_cols=None,
        scalar_type="min_max",  # 'min_max' or 'standard
        dequantize=False,
        seed=1,
        is_range_queries=True,
        verbose=False,
        enable_query_dequantize=True
    )

    # load error compensation model
    error_comp_model_path = get_model_path(dataset_type.value) / "error_comp_model.pt"
    error_comp_model = (
        ErrorCompensationNetwork(error_comp_model_path, dataset)
        if not IS_ERROR_COMP_TRAIN
        else None
    )

    df = dataset.df
    no_of_rows = df.shape[0]
    min_values_list = df.min().values
    max_values_list = df.max().values
    logger.info(f"Columns: {df.columns}")

    new_query_l = []
    new_query_r = []
    actual_ce = []

    query_l = dataset.query_l[:, COLUMN_INDEXES]
    query_r = dataset.query_r[:, COLUMN_INDEXES]
    actual_ce_ds = dataset.true_card

    loop = tqdm(enumerate(zip(query_l, query_r, actual_ce_ds)), total=query_l.shape[0])
    query_size = 0
    new_query_l = query_l
    new_query_r = query_r
    actual_ce = actual_ce_ds

    logger.info(f"Query Size: {len(new_query_l)}")

    def get_query(q1, q2):
        return np.array([[q11, q22] for q11, q22 in zip(q1, q2)]).reshape(-1)

    X = np.array([get_query(ql, qr) for ql, qr in zip(new_query_l, new_query_r)])
    y = np.array(actual_ce) / no_of_rows

    cdf_df = CDFStorage(
        OptimizedEmpiricalCDFModel,
        cached_name_string=CDF_STORAGE_CACHE,
        override=CDF_STORAGE_CACHE_OVERRIDE,
        emp_method=EMPMethod.RELATIVE,
        enable_low_precision=False,
        max_unique_values="auto",
    )
    cdf_df.fit(df)

    if df.shape[0] > 25_000_000:
        """Take a sample of 20_000_000 rows"""
        begin_time_sampling = time.time()
        data_np = (
            df.sample(n=20_000_000, random_state=1, replace=False)
            .to_numpy()
            .transpose()
        )
        logger.info(f"Time Taken for Sampling: {time.time() - begin_time_sampling}")
    else:
        data_np = df.to_numpy().transpose()

    theta_dict = ThetaStorage(COPULA_TYPE, NO_OF_COLUMNS).get_theta(
        data_np, cache_name=THETA_STORAGE_CACHE
    )

    model = DivineCopulaDynamicRecursive(theta_dict=theta_dict)

    # model.verbose = True
    full_zero_count = 0
    nan_count = 0

    dict_list = []
    time_taken_list = []
    time_taken_predict_cdf_list = []
    loop = tqdm(zip(X, y), total=X.shape[0])
    for query, y_act in loop:
        query = query.reshape(1, -1)
        # start_time_cdf = time.time()
        start_time_predict_cdf = time.time()
        cdf_list = cdf_df.get_converted_cdf(query, COLUMN_INDEXES, nproc=1, cache=False)
        time_taken_predict_cdf = time.time() - start_time_predict_cdf
        # time_taken_cdf = time.time() - start_time_cdf
        # Reshape the array into pairs
        reshaped_cdf_list = cdf_list.reshape(-1, 2)
        # Identifying the indexes where the first value is not equal to 0 and the second value is not equal to 1
        non_zero_non_one_indices = np.where(
            (reshaped_cdf_list[:, 0] != 0) | (reshaped_cdf_list[:, 1] != 1)
        )[0]
        col_indices = [i + 1 for i in non_zero_non_one_indices]
        cdf_list = reshaped_cdf_list[non_zero_non_one_indices].reshape(-1)
        no_of_cols_for_this_query = len(col_indices)
        loop.set_description(f"#cols: {no_of_cols_for_this_query:2d}")
        y_bar = None

        start_time = time.time()
        y_bar = (
            model.predict(cdf_list, column_list=col_indices) if y_bar is None else y_bar
        )
        time_taken = time.time() - start_time

        time_taken_list.append(time_taken)
        time_taken_predict_cdf_list.append(time_taken_predict_cdf)
        # if len(time_taken_list) == 1000:
        #     break

        if np.isnan(y_bar):  # or np.isnan(y_bar_2):
            nan_count += 1
            continue
        q_error = qerror(y_bar, y_act, no_of_rows=no_of_rows)

        if not IS_ERROR_COMP_TRAIN:
            y_bar_2 = error_comp_model.inference(
                query=query, cdf=cdf_list, y_bar=y_bar
            )[0]
            q_error_2 = qerror(y_bar_2, y_act, no_of_rows=None)
        else:
            q_error_2 = None
            y_bar_2 = None

        dict_list.append(
            {
                "X": ",".join(list(map(str, cdf_list))),
                "query": ",".join(list(map(str, query[0]))),
                "y_bar": y_bar,
                "y_bar_2": y_bar_2,
                "y": y_act,
                "gt": y_act * no_of_rows,
                "y_bar_card": max(y_bar * no_of_rows, 1),
                "y_card": y_act * no_of_rows,
                "time_taken": time_taken,
                "q_error": q_error,
                "q_error_2": q_error_2,
                "exec_count": model.exec_count,
            }
        )

    # percentiles_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    # for percentile in percentiles_values:
    #     value = np.percentile(time_taken_list, percentile)
    #     logger.info(f"Time Percentile ({percentile}th): {value}")

    logger.info(f"Time Taken: {np.average(time_taken_list) * 1000} ms")
    logger.info(
        f"Time Taken Predict CDF: {np.average(time_taken_predict_cdf_list) * 1000} ms"
    )

    df1 = pd.DataFrame(dict_list)

    dict_list = []
    percentiles_values = [50, 90, 95, 99, 100]
    logger.info("-" * 40)
    logger.info(f"Percentiles for q_error")
    for percentile in percentiles_values:
        value = np.percentile(df1["q_error"], percentile)
        value_2 = (
            np.percentile(df1["q_error_2"], percentile)
            if not IS_ERROR_COMP_TRAIN
            else None
        )

        value_1_str = f"{value:.3f}"
        value_2_str = f"{value_2:.3f}" if value_2 is not None else "N/A"

        if IS_ERROR_COMP_TRAIN:
            logger.info(f"Percentile ({percentile:3d}th): {value_1_str}")
            dict_list.append({"percentile": percentile, "value": value_1_str})
        else:
            dict_list.append(
                {"percentile": percentile, "before": value, "after": value_2}
            )
        logger.info(
            f"Percentile ({percentile:3d}th): copula_only: {value_1_str:>10}, copula+error_comp: {value_2_str:>10}"
        )

    dict_list.append({"percentile": "", "value": ""})
    dict_list.append({"percentile": "NO_OF_ROWS", "value": NO_OF_ROWS})
    dict_list.append({"percentile": "QUERY_SIZE", "value": QUERY_SIZE})

    df2 = pd.DataFrame(dict_list)

    with pd.ExcelWriter(EXCEL_FILE_PATH, mode="w") as writer:
        df1.to_excel(writer, sheet_name="Results")
        df2.to_excel(writer, sheet_name="Percentiles")

    logger.info("-" * 40)
    logger.info(f"Query Size: {df1.shape[0]}")
    logger.info(f"Full Zero Count: {full_zero_count}")
    logger.info(f"NaN Count: {nan_count}")


if __name__ == "__main__":
    main()
