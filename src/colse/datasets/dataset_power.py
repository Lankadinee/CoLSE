import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from colse.cat_transform import DeqDataTypes, DeQuantize
from colse.data_path import get_data_path
from colse.datasets.params import ROW_PREFIX
from colse.df_utils import load_dataframe

current_dir = Path(__file__).parent
# dataset_dir = current_dir.joinpath("../../data/power")
dataset_dir = get_data_path() / "power/"
IS_DEQUANTIZE = True


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", 500_000)
    no_of_columns = kwargs.get("no_of_cols", None)
    selected_cols = kwargs.get("selected_cols", None)
    data_file_name = kwargs.get("data_file_name", "original.csv")
    if data_file_name is None:
        data_file_name = "original.csv"

    """ Load dataset"""
    # df_path = dataset_dir / (
    #     "original_dequantized_all.csv" if IS_DEQUANTIZE else "original.csv"
    # )
    # df_path = dataset_dir / ("original_dequantized_v2.parquet" if IS_DEQUANTIZE else "original.csv")
    df_path = dataset_dir / data_file_name

    logger.info(f"Loading power dataset from: {df_path}")
    df = load_dataframe(df_path)
    # df = df[df['Voltage'] > 200]
    # df.dropna(inplace=True)

    nrows = df.shape[0] if nrows is None else nrows

    attr1 = df["Global_active_power"].to_numpy()[:nrows]
    attr2 = df["Global_reactive_power"].to_numpy()[:nrows]
    attr3 = df["Voltage"].to_numpy()[:nrows]
    attr4 = df["Global_intensity"].to_numpy()[:nrows]
    attr5 = df["Sub_metering_1"].to_numpy()[:nrows]
    attr6 = df["Sub_metering_2"].to_numpy()[:nrows]
    attr7 = df["Sub_metering_3"].to_numpy()[:nrows]
    # data = df.to_numpy().astype(int)

    # Stack the attributes into a 2D array
    data = np.column_stack((attr1, attr2, attr3, attr4, attr5, attr6, attr7))

    new_df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 8)])

    for new_col, old_col in zip(new_df.columns, df.columns):
        new_df[new_col] = new_df[new_col].astype(df[old_col].dtype)

    if no_of_columns:
        new_df = new_df.iloc[:, :no_of_columns]

    if selected_cols:
        new_df = new_df.iloc[:, selected_cols]

    # df = df.astype(np.float64)
    return new_df


def get_queries_power(**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    data_split = kwargs.get("data_split", "train")
    no_of_queries = kwargs.get("no_of_queries", None)
    min_value = kwargs.get("min_value", 1)
    type_test = kwargs.get("type_test", False)
    is_test_set = kwargs.get("is_test_set", False)
    enable_query_dequantize = kwargs.get("enable_query_dequantize", False)

    """Load queries"""
    query_json = dataset_dir.joinpath("query_power7.json")
    logger.info(f"Loading queries from {query_json.absolute()}")
    queries = json.load(query_json.open())
    # training_queries = queries['train']
    # validation_queries = queries['valid']
    # testing_queries = queries['test']

    query_l = []
    query_r = []
    for query in queries[data_split]:
        query = query[0]
        lb_list = []
        ub_list = []
        for key in query.keys():
            if isinstance(query[key], list) and query[key][0] == "[]":
                lb_list.append(query[key][1][0])
                ub_list.append(query[key][1][1])
            elif isinstance(query[key], list) and query[key][0] == "<=":
                lb_list.append(-np.inf)
                ub_list.append(query[key][1])
            elif isinstance(query[key], list) and query[key][0] == ">=":
                lb_list.append(query[key][1])
                ub_list.append(np.inf)
            elif isinstance(query[key], list) and query[key][0] == "=":
                lb_list.append(query[key][1])
                ub_list.append(query[key][1])
            else:
                lb_list.append(-np.inf)
                ub_list.append(np.inf)

        query_l.append(np.array(lb_list))
        query_r.append(np.array(ub_list))

    "Load true cardinality"
    labels = pd.read_csv(dataset_dir.joinpath(f"label_power7_{data_split}.csv"))
    true_card = labels["cardinality"].to_numpy().astype(int)

    # if data_split == "train":
    #     #     """Remove very small values"""
    #     #     # true_card = np.where(true_card < 1, 1, true_card)
    #     #     # check for the indices where the true_card is less than 1 and remove them
    #     indices = np.where(true_card == 0)[0]
    #     query_l = np.delete(query_l, indices, axis=0)
    #     query_r = np.delete(query_r, indices, axis=0)
    #     true_card = np.delete(true_card, indices, axis=0)

    if no_of_queries is not None:
        query_l = np.array(query_l[:no_of_queries])
        query_r = np.array(query_r[:no_of_queries])
        true_card = true_card[:no_of_queries]
    else:
        """convert all the data into float64"""
        query_l = np.array(query_l)
        query_r = np.array(query_r)
        true_card = true_card

    if enable_query_dequantize:
        logger.info("Dequantizing queries...")
        query_l_new, query_r_new = query_value_mapper(query_l, query_r)
        return (
            query_l_new.astype(np.float64),
            query_r_new.astype(np.float64),
            true_card.astype(np.float64),
        )

    return query_l, query_r, true_card


# def query_value_mapper(query_l, query_r):
#     df = load_dataframe("library/data/power/original.csv")
#     dequantize = DeQuantize()
#     for col in range(query_l.shape[1]):
#         mapping = dequantize.fit(df.iloc[:, col].to_numpy()).mapping
#         for q_l, q_r in zip(query_l, query_r):
#             if q_l[col] == q_r[col]:
#                 q_l[col] = mapping[q_l[col]][0]
#                 q_r[col] = mapping[q_r[col]][1]


#     return query_l, query_r
def query_value_mapper(query_l, query_r):
    df = load_dataframe(dataset_dir / "original.csv")

    quant_dict = DeQuantize.get_dequantizable_columns(
        df, col_list_to_be_dequantized=list(df.columns)
    )
    loop = tqdm(enumerate(list(df.columns)), total=len(df.columns))
    for col_id, col_name in loop:
        dequantize = DeQuantize()
        dequantize.fit(df[col_name].to_numpy())
        if quant_dict[col_name].is_dequantizable:
            loop.set_description(f"Mapping values > {col_name:25}")
            if quant_dict[col_name].data_type == DeqDataTypes.DISCRETE:
                for q_l, q_r in zip(query_l, query_r):
                    q_l[col_id] = (
                        dequantize.get_mapping(q_l[col_id])
                        if q_l[col_id] != -np.inf
                        else -np.inf
                    )
                    q_r[col_id] = (
                        dequantize.get_mapping(q_r[col_id])
                        if q_r[col_id] != np.inf
                        else np.inf
                    )
            else:
                raise ValueError(
                    f"Data type {quant_dict[col_name].data_type} not supported"
                )

    return query_l, query_r


# def query_value_mapper(query_l, query_r):
#     df = load_dataframe(dataset_dir / "original.csv")

#     quant_dict = DeQuantize.get_dequantizable_columns(
#         df, col_list_to_be_dequantized=list(df.columns)
#     )
#     dequantize = SplineDequantizer()
#     dequantize.fit(df, columns=list(df.columns))
#     loop = tqdm(enumerate(list(df.columns)), total=len(df.columns))
#     for col_id, col_name in loop:
#         # dequantize = DeQuantize()
#         if quant_dict[col_name].is_dequantizable:
#             loop.set_description(f"Mapping values > {col_name:25}")
#             if quant_dict[col_name].data_type == DeqDataTypes.DISCRETE:
#                 for q_l, q_r in zip(query_l, query_r):
#                     q_l[col_id] = (
#                         # dequantize.get_mapping(q_l[col_id])
#                         dequantize.get_continuous_interval(col_name, q_l[col_id])[0]
#                         if q_l[col_id] != -np.inf
#                         else -np.inf
#                     )
#                     q_r[col_id] = (
#                         # dequantize.get_mapping(q_r[col_id])
#                         dequantize.get_continuous_interval(col_name, q_r[col_id])[1]
#                         if q_r[col_id] != np.inf
#                         else np.inf
#                     )
#             else:
#                 raise ValueError(
#                     f"Data type {quant_dict[col_name].data_type} not supported"
#                 )

#     return query_l, query_r


if __name__ == "__main__":
    query_l, query_r, true_card = get_queries_power(data_split="test")
    # print(query_l.shape)
    # print(query_r.shape)
    # print(true_card.shape)
    # df = generate_dataset()
    # print(df.head())
    # print(_df.shape)
    # print(dfs.head())
    # print(dfs.shape)
