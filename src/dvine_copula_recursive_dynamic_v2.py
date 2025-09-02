import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from colse.column_index import ColumnIndexProvider, ColumnIndexTypes
from colse.copula_types import CopulaTypes
from colse.custom_data_generator import CustomDataGen
from colse.data_conversion_params import DataConversionParams
from colse.data_path import DataPathDir, get_data_path, get_log_path, get_model_path
from colse.dataset_names import DatasetNames
from colse.datasets.preprocess import preprocess_dataset
from colse.df_utils import load_dataframe
from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive
from colse.fanout_scaling import FanoutScaling
from colse.q_error import qerror
from colse.spline_dequantizer import SplineDequantizer
from colse.theta_storage import ThetaStorage
from error_comp_network import ErrorCompensationNetwork

current_dir = Path(__file__).resolve().parent
iso_time_str = datetime.now().isoformat()
iso_time_str = iso_time_str.replace(":", "-")
logs_dir = get_log_path()
pp_enb = True
logger.add(
    logs_dir.joinpath(f"colse-{iso_time_str}.log"),
    rotation="10 MB",
    level="DEBUG",
)
SHOW_DEBUG_INFO = False
# Set a global random seed for reproducibility
SEED = 42
np.random.seed(SEED)
ENABLE_DEQUANTIZER_UNIQUES_SHUFFLING = False


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
    parser.add_argument(
        "--model_name", type=str, default=None, help="Name of the model"
    )
    parser.add_argument(
        "--update_type", type=str, default=None, help="Type of update to the dataset"
    )
    parser.add_argument(
        "--workload_updates", type=str, default=None, help="Name of the workload updates"
    )
    parser.add_argument(
        "--output_excel_name", type=str, default=None, help="Name of the output excel file"
    )
    parser.add_argument(
        "--theta_cache_path", type=str, default="theta_cache.pkl", help="Path to the theta cache file"
    )
    parser.add_argument(
        "--cdf_cache_name", type=str, default="cdf_cache.pkl", help="Name of the cdf cache file"
    )
    parser.add_argument(
        "--no_of_training_queries", type=int, default=0, help="Number of training queries"
    )
    parser.add_argument(
        "--sparcity", type=float, default=1.0, help="Sparcity of the queries"
    )
    return parser.parse_args()



def show_args(parsed_args):
    table = Table(title="Arguments to the script")
    table.add_column("Argument", justify="left")
    table.add_column("Value", justify="right")
    for arg, value in parsed_args.__dict__.items():
        table.add_row(arg, str(value))
    table.add_row("-"*10, "-"*10)
    table.add_row("Pre-processing", "True" if pp_enb else "False")
    console = Console()
    console.print(table)



def main():
    parsed_args = parse_args()
    show_args(parsed_args)
    data_split = parsed_args.data_split
    dataset_type = DatasetNames(parsed_args.dataset_name)
    logger.info(f"Dataset Type: {parsed_args.dataset_name} -> {dataset_type.name}")

    NO_OF_ROWS = None
    QUERY_SIZE = None
    COLUMN_INDEXES = None
    NO_OF_COLUMNS = dataset_type.get_no_of_columns()

    # pre process dataset
    preprocess_dataset(dataset_type, skip_if_exists=True, pp_enb=pp_enb)
    
    error_comp_model = None
    error_comp_model_path = None
    if parsed_args.model_name:
        logger.info(f"Loading error compensation model {parsed_args.model_name}")
        # load error compensation model
        error_comp_model_path = get_model_path(dataset_type.value) / f"{parsed_args.model_name}"
        if not error_comp_model_path.exists():
            logger.error(f"Error compensation model {error_comp_model_path} does not exist")
            exit(1)
        
    logger.info(f"Error compensation model loaded: {error_comp_model is not None}")

    excel_file_path = get_data_path(f"{DataPathDir.EXCELS}/{dataset_type.value}") / f"{parsed_args.output_excel_name}"
    COPULA_TYPE = CopulaTypes.GUMBEL
    theta_cache_path = get_data_path(DataPathDir.THETA_CACHE, dataset_type.value) / f"{parsed_args.theta_cache_path}"

    # Dequantize dataset
    if parsed_args.update_type:
        original_file_name = f"{DataPathDir.DATA_UPDATES}/original_{parsed_args.update_type}.csv"
        output_file_name = f"{DataPathDir.DATA_UPDATES}/dequantized_v2_{parsed_args.update_type}.parquet"
        query_file_name = f"{DataPathDir.DATA_UPDATES}/query_{parsed_args.update_type}.json"
    else:
        original_file_name = dataset_type.get_file_path(pp_enb=pp_enb)
        output_file_name = "dequantized_v2.parquet"
        if parsed_args.workload_updates:
            query_file_name = f"{DataPathDir.WORKLOAD_UPDATES}/query_{parsed_args.workload_updates}.json"
        else:
            query_file_name = None

    s_dequantize = SplineDequantizer(
        dataset_type=dataset_type,
        cache_name=parsed_args.cdf_cache_name,
        output_file_name=output_file_name,
        enable_uniques_shuffling=ENABLE_DEQUANTIZER_UNIQUES_SHUFFLING
    )
    s_dequantize.fit_transform(load_dataframe(dataset_type.get_file_path(original_file_name)))
    
    # this will be non if there are no non-continuous columns
    dequantized_file_name = s_dequantize.get_dequantized_dataset_name()
    if dequantized_file_name:
        # there are non-continuous columns, so we use the converted file
        datagen_load_file_name = dequantized_file_name
    else:
        datagen_load_file_name = original_file_name

    logger.info(f"Loading dataset from {datagen_load_file_name}")
    logger.info(f"Query file name: {query_file_name}")
    dataset = CustomDataGen(
        no_of_rows=NO_OF_ROWS,
        no_of_queries=None,
        data_file_name=datagen_load_file_name,
        query_file_name=query_file_name,
        dataset_type=dataset_type,
        data_split=data_split,
        selected_cols=None,
        scalar_type="min_max",  # 'min_max' or 'standard
        dequantize=False,
        seed=1,
        is_range_queries=True,
        verbose=False,
        enable_query_dequantize=False,
    )

    dc_params = DataConversionParams(dataset_type, parsed_args.update_type)
    dc_params.store_data_conversion_params(dataset)

    if error_comp_model_path:
        # TODO - Send dc_params to the error compensation model instead of dataset
        error_comp_model = ErrorCompensationNetwork(error_comp_model_path, dataset)

    df = dataset.df
    no_of_rows = df.shape[0]
    logger.info(f"No of rows: {no_of_rows}")
    logger.info(f"Columns: {df.columns}")

    new_query_l = []
    new_query_r = []
    actual_ce = []

    # query_l = dataset.query_l[:, COLUMN_INDEXES]
    # query_r = dataset.query_r[:, COLUMN_INDEXES]
    # actual_ce_ds = dataset.true_card
    query_l, query_r, actual_ce_ds = dataset.get_queries(sparcity=parsed_args.sparcity)

    if parsed_args.update_type and data_split == "train":
        no_of_rows = 8000
        idx = np.random.choice(len(query_l), no_of_rows, replace=False)
        new_query_l = query_l[idx]
        new_query_r = query_r[idx]
        actual_ce = actual_ce_ds[idx]
    else:
        new_query_l = query_l
        new_query_r = query_r
        actual_ce = actual_ce_ds

    current_no_of_rows = len(new_query_l)
    no_of_training_queries =  parsed_args.no_of_training_queries
    if no_of_training_queries and no_of_training_queries < current_no_of_rows:
        new_query_l = new_query_l[:no_of_training_queries]
        new_query_r = new_query_r[:no_of_training_queries]
        actual_ce = actual_ce[:no_of_training_queries]
        logger.info(f"Reducing the number of training queries from {current_no_of_rows} to {no_of_training_queries}")

    logger.info(f"Query Size: {len(new_query_l)}")

    def get_query(q1, q2):
        return np.array([[q11, q22] for q11, q22 in zip(q1, q2)]).reshape(-1)

    X = np.array([get_query(ql, qr) for ql, qr in zip(new_query_l, new_query_r)])
    y = np.array(actual_ce) / no_of_rows

    theta_dict = ThetaStorage(COPULA_TYPE, NO_OF_COLUMNS).get_theta(
        df, cache_name=theta_cache_path
    )

    col_index_provider = ColumnIndexProvider(df, ColumnIndexTypes.NATURAL_SKIP_ORDERING)
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
        
        start_time_predict_cdf = time.time()
        original_cdf_list = s_dequantize.get_converted_cdf(query, COLUMN_INDEXES)
        time_taken_predict_cdf = time.time() - start_time_predict_cdf

        # logger.info(f"Query [{query.shape}]: {query}")
        # logger.info(f"Original CDF [{original_cdf_list.shape}]: {original_cdf_list}")
        # logger.info(f"Query Modified [{query_modified.shape}]: {query_modified}")
        # logger.info(f"CDF [{cdf_list.shape}]: {cdf_list}")

        col_indices, cdf_list = col_index_provider.get_column_index(original_cdf_list)
        no_of_cols_for_this_query = len(col_indices)
        loop.set_description(f"#cols: {no_of_cols_for_this_query:2d}")
        start_time = time.time()
        y_bar = model.predict(cdf_list, column_list=col_indices)
        copula_pred_time = time.time() - start_time

        time_taken_list.append(copula_pred_time)
        time_taken_predict_cdf_list.append(time_taken_predict_cdf)
        

        if np.isnan(y_bar):  # or np.isnan(y_bar_2):
            nan_count += 1
            continue
        q_error = qerror(y_bar, y_act, no_of_rows=no_of_rows)
        mapped_query = s_dequantize.get_mapped_query(query, COLUMN_INDEXES)
        # logger.info(f"Prediction: {y_bar} Mapped query: {mapped_query}")
        if not any(mapped_query):
            logger.warning(f"Mapped query is empty for query: {query}")
            raise ValueError(f"Mapped query is empty for query: {query}")

        if error_comp_model:
            if y_bar == 0:
                y_bar_2 = 0
                q_error_2 = q_error
                # logger.warning(f"y_bar is 0 actual[{y_act}] Error:{q_error} for query: {query_modified},\n CDF: {cdf_list}\n Mapped query: {mapped_query}\n skipping error compensation")
            else:
                y_bar_2 = error_comp_model.inference(
                    query=mapped_query, cdf=cdf_list, y_bar=y_bar
                )[0]
                q_error_2 = qerror(y_bar_2, y_act, no_of_rows=None)
        else:
            q_error_2 = None
            y_bar_2 = None

        #####################################################
        if SHOW_DEBUG_INFO:
            # query_modified = query.reshape(-1,2)[non_zero_non_one_indices]    
            table = Table(title="Debug Info")
            table.add_column("Item", justify="right")
            table.add_column("Shape", justify="right")
            table.add_column("Value", justify="right")

            table.add_row("Query", f"{query.shape}", f"{query}")
            table.add_row("Original CDF", f"{original_cdf_list.shape}", f"{original_cdf_list.round(3)}")
            # table.add_row("Modified Query", f"{query_modified.shape}", f"{query_modified}")
            table.add_row("CDF", f"{cdf_list.shape}", f"{cdf_list}")
            table.add_row("Mapped Query", f"{mapped_query.shape}", f"{mapped_query}")
            table.add_row("Y Bar", f"{y_bar.shape}", f"{y_bar}")
            table.add_row("Y Bar 2", f"{y_bar_2.shape if y_bar_2 is not None else 'None'}", f"{y_bar_2}")
            table.add_row("Y", f"{y_act.shape if y_act is not None else 'None'}", f"{y_act}")
            console = Console()

            console.print(table)

        #####################################################

        dict_list.append(
            {
                "X": ",".join(list(map(str, cdf_list))),
                "query": ",".join(list(map(str, query[0]))),
                "mapped_query": ",".join(list(map(str, mapped_query))),
                "y_bar": y_bar,
                "y_bar_2": y_bar_2,
                "y": y_act,
                "gt": y_act * no_of_rows,
                "y_bar_card": max(y_bar * no_of_rows, 1),
                "y_card": y_act * no_of_rows,
                "time_taken": copula_pred_time,
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
    logger.info("Percentiles for q_error")
    table = Table(title="Dequantizer Test")
    table.add_column("Percentile", justify="right")
    table.add_column("copula", justify="right")
    if error_comp_model:
        table.add_column("copula+error_comp", justify="right")

    for percentile in percentiles_values:
        value = np.percentile(df1["q_error"], percentile)
        value_1_str = f"{value:.3f}"

        if error_comp_model:
            value_2 = np.percentile(df1["q_error_2"], percentile)
            value_2_str = f"{value_2:.3f}"
            dict_list.append(
                {"percentile": percentile, "before": value, "after": value_2}
            )
            table.add_row(f"{percentile}", f"{value_1_str}", f"{value_2_str}")
        else:

            dict_list.append({"percentile": percentile, "value": value_1_str})
            table.add_row(f"{percentile}", f"{value_1_str}")

    console = Console()
    console.print(table)

    logger.info("Saving results to Excel file")
    dict_list.append({"percentile": "", "value": ""})
    dict_list.append({"percentile": "NO_OF_ROWS", "value": NO_OF_ROWS})
    dict_list.append({"percentile": "QUERY_SIZE", "value": QUERY_SIZE})

    df2 = pd.DataFrame(dict_list)

    with pd.ExcelWriter(excel_file_path, mode="w") as writer:
        df1.to_excel(writer, sheet_name="Results")
        df2.to_excel(writer, sheet_name="Percentiles")
    logger.info(f"Saved results to {excel_file_path}")

    logger.info("-" * 40)
    logger.info(f"Query Size: {df1.shape[0]}")
    logger.info(f"Full Zero Count: {full_zero_count}")
    logger.info(f"NaN Count: {nan_count}")
    logger.info("-" * 40)
    logger.info("Done!\n\n")


if __name__ == "__main__":
    main()
