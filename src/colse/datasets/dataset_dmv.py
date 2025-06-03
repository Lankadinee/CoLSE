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
from colse.spline_dequantizer import SplineDequantizer

current_dir = Path(__file__).parent
dataset_dir = get_data_path() / "dmv/"
IS_DEQUANTIZE = True
# MIN_DATE = '1992-01-02'


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", 500_000)
    no_of_columns = kwargs.get("no_of_cols", None)
    selected_cols = kwargs.get("selected_cols", None)

    logger.info("Loading dmv dataset...")

    df_path = dataset_dir / (
        "dmv_dequantized_v2.parquet" if IS_DEQUANTIZE else "dmv.csv"
    )
    df = load_dataframe(df_path)
    logger.info("DMV dataframe loaded.")

    nrows = df.shape[0] if nrows is None else nrows

    # if IS_DEQUANTIZE:

    #     df["Reg_Valid_Date"] = pd.to_datetime(df["Reg_Valid_Date"], format="%Y-%m-%d")

    #     """find minimum date and subtract from all dates"""
    #     min_date = min(df["Reg_Valid_Date"].min())
    #     print(min_date)

    #     df["Reg_Valid_Date"] = (df["Reg_Valid_Date"] - min_date).dt.days

    logger.info(f"Convert the first {nrows} rows to numpy array")
    attr1 = df["Record_Type"].to_numpy()[:nrows]
    attr2 = df["Registration_Class"].to_numpy()[:nrows]
    attr3 = df["State"].to_numpy()[:nrows]
    attr4 = df["County"].to_numpy()[:nrows]
    attr5 = df["Body_Type"].to_numpy()[:nrows]
    attr6 = df["Fuel_Type"].to_numpy()[:nrows]
    attr7 = df["Reg_Valid_Date"].to_numpy()[:nrows]
    attr8 = df["Color"].to_numpy()[:nrows]
    attr9 = df["Scofflaw_Indicator"].to_numpy()[:nrows]
    attr10 = df["Suspension_Indicator"].to_numpy()[:nrows]
    attr11 = df["Revocation_Indicator"].to_numpy()[:nrows]
    # data = df.to_numpy().astype(int)

    logger.info("Stacking the attributes into a 2D array")
    # Stack the attributes into a 2D array
    data = np.column_stack(
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10, attr11)
    )

    new_df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 12)])

    for new_col, old_col in zip(new_df.columns, df.columns):
        new_df[new_col] = new_df[new_col].astype(df[old_col].dtype)

    if no_of_columns:
        new_df = new_df.iloc[:, :no_of_columns]

    if selected_cols:
        new_df = new_df.iloc[:, selected_cols]

    # df = df.astype(np.float64)
    return new_df


def get_queries_dmv(**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading DMV queries...")

    data_split = kwargs.get("data_split", "train")
    no_of_queries = kwargs.get("no_of_queries", None)
    min_value = kwargs.get("min_value", 1)
    type_test = kwargs.get("type_test", False)
    is_test_set = kwargs.get("is_test_set", False)

    """Load queries"""
    query_json = dataset_dir / "query_dmv11.json"
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
    labels = load_dataframe(dataset_dir / f"label_dmv11_{data_split}.csv")
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

    if IS_DEQUANTIZE:
        """convert to np float 64"""
        logger.info("Dequantizing the queries...")
        # query_l = query_l.astype(np.float64)
        # query_r = query_r.astype(np.float64)
        query_l, query_r = query_value_mapper(query_l, query_r)
        return (
            query_l.astype(np.float64),
            query_r.astype(np.float64),
            true_card.astype(np.float64),
        )

    return query_l, query_r, true_card


def query_value_mapper_bk(query_l, query_r):
    df = load_dataframe(dataset_dir / "dmv.csv")

    quant_dict = DeQuantize.get_dequantizable_columns(
        df, col_list_to_be_dequantized=None
    )
    all_cols = len(list(df.columns))
    loop = tqdm(enumerate(list(df.columns)), total=all_cols)
    for col_id, col_name in loop:
        loop.set_description(f"Mapping values > {col_name:25}")
        dequantize = DeQuantize()
        mapping = dequantize.fit(df.iloc[:, col_id].to_numpy()).mapping
        if quant_dict[col_name].is_dequantizable:
            if quant_dict[col_name].data_type == DeqDataTypes.CATEGORICAL:
                for q_l, q_r in zip(query_l, query_r):
                    if q_l[col_id] == q_r[col_id]:
                        q_l[col_id] = mapping[q_l[col_id]][0]
                        q_r[col_id] = mapping[q_r[col_id]][1]
            elif quant_dict[col_name].data_type == DeqDataTypes.DISCRETE:
                for q_l, q_r in zip(query_l, query_r):
                    q_l[col_id] = (
                        dequantize.get_mapping(q_l[col_id])
                        if q_l[col_id] not in [-np.inf, "-inf"]
                        else -np.inf
                    )
                    q_r[col_id] = (
                        dequantize.get_mapping(q_l[col_id])
                        if q_r[col_id] not in [np.inf, "inf"]
                        else np.inf
                    )
            else:
                raise ValueError(
                    f"Data type {quant_dict[col_name].data_type} not supported"
                )

    return query_l, query_r


# New method
def query_value_mapper(query_l, query_r):
    df = load_dataframe(dataset_dir / "dmv.csv")

    quant_dict = DeQuantize.get_dequantizable_columns(
        df, col_list_to_be_dequantized=None
    )
    all_cols = list(df.columns)
    all_cols_len = len(all_cols)

    dequantizer = SplineDequantizer(M=10000)
    dequantizer.fit(df, columns=all_cols)

    loop = tqdm(enumerate(all_cols), total=all_cols_len)
    for col_id, col_name in loop:
        loop.set_description(f"Mapping values v2 > {col_name:25}")

        if quant_dict[col_name].is_dequantizable:
            if quant_dict[col_name].data_type == DeqDataTypes.CATEGORICAL:
                for q_l, q_r in zip(query_l, query_r):
                    if q_l[col_id] == q_r[col_id]:
                        q_l[col_id], q_r[col_id] = dequantizer.get_interval(
                            col_name, q_l[col_id]
                        )
                    # else:
                    #     logger.error(f"Query value {q_l[col_id]} and {q_r[col_id]} are not equal")
                    #     raise ValueError(f"Query value {q_l[col_id]} and {q_r[col_id]} are not equal")

            elif quant_dict[col_name].data_type == DeqDataTypes.DISCRETE:
                for q_l, q_r in zip(query_l, query_r):
                    q_l[col_id] = (
                        dequantizer.get_interval(q_l[col_id])[0]
                        if q_l[col_id] not in [-np.inf, "-inf"]
                        else -np.inf
                    )
                    q_r[col_id] = (
                        dequantizer.get_interval(q_l[col_id])[1]
                        if q_r[col_id] not in [np.inf, "inf"]
                        else np.inf
                    )
            else:
                raise ValueError(
                    f"Data type {quant_dict[col_name].data_type} not supported"
                )

    return query_l, query_r


def queried_values(query_list):
    """Get the queried values"""
    query_dict = {}
    for i in range(query_list.shape[0]):
        for j in range(query_list[i].shape[0]):
            if query_list[i][j] != -np.inf:
                query_dict[j] = query_dict.get(j, []) + [query_list[i][j]]
    return query_dict


def get_sample_queries(no_of_queries, queried_columns: list = None, no_inf=True):
    query_l, query_r, true_card = get_queries_dmv()
    unique_values_dict = dict()
    no_of_cols = query_l.shape[1]
    for col_index in range(no_of_cols):
        query_l_unique = np.unique(query_l[:, col_index])
        query_r_unique = np.unique(query_r[:, col_index])
        combined_unique = np.union1d(query_l_unique, query_r_unique)
        """remove -inf and +inf values"""
        combined_unique = combined_unique[~np.isinf(combined_unique)]
        unique_values_dict[col_index] = combined_unique

    if queried_columns is None:
        queried_columns = list(unique_values_dict.keys())

    query_l = np.array([])
    query_r = np.array([])
    selected_col_indexes = []
    for col_index in range(no_of_cols):
        if col_index in queried_columns:
            col_sample = np.array(
                [
                    np.random.choice(unique_values_dict[col_index], no_of_queries)
                    for _ in range(2)
                ]
            )
            col_sample_sorted = np.sort(col_sample, axis=0)
            ub_queries, lb_queries = col_sample_sorted[1, :], col_sample_sorted[0, :]
        else:
            if no_inf:
                continue
            ub_queries, lb_queries = np.array([+np.inf] * no_of_queries), np.array(
                [-np.inf] * no_of_queries
            )
        selected_col_indexes.append(col_index)
        query_l = np.vstack((query_l, lb_queries)) if query_l.size else lb_queries
        query_r = np.vstack((query_r, ub_queries)) if query_r.size else ub_queries
    logger.info(f"Selected columns: {selected_col_indexes}")
    return query_l.transpose(), query_r.transpose()


if __name__ == "__main__":
    # indices = [0, 1, 515, 6, 523, 14, 528, 530, 531, 24, 536, 27, 541, 543, 548, 549, 39, 40, 42, 554, 555, 556, 565,
    #            578, 74, 592, 86, 599, 89, 604, 605, 95, 98, 103, 104, 616, 111, 625, 627, 628, 629, 630, 120, 126, 640,
    #            129, 132, 135, 136, 141, 144, 658, 659, 148, 149, 663, 665, 155, 667, 668, 161, 167, 168, 169, 172, 175,
    #            689, 179, 184, 698, 703, 706, 709, 199, 200, 206, 718, 210, 725, 729, 734, 228, 741, 742, 231, 234, 242,
    #            246, 771, 775, 264, 781, 783, 272, 273, 784, 787, 278, 795, 289, 292, 293, 300, 302, 304, 305, 817, 308,
    #            820, 828, 318, 319, 834, 835, 324, 325, 328, 840, 841, 844, 333, 847, 849, 850, 341, 342, 854, 859, 860,
    #            350, 365, 367, 881, 892, 390, 393, 906, 907, 909, 398, 399, 913, 403, 917, 408, 925, 415, 416, 422, 940,
    #            945, 441, 443, 955, 446, 959, 448, 450, 962, 964, 966, 455, 968, 460, 461, 975, 464, 465, 466, 980, 474,
    #            476, 484, 997, 488, 497, 502, 510]
    # df = generate_dataset(nrows=None)
    query_l, query_r, true_card = get_queries_dmv()
    query_l_new, query_r_new = query_value_mapper(query_l, query_r)
    print(query_l_new)
    # queried_values = queried_values(query_l)
    # values_dict = get_sample_queries(queried_columns=[2, 3])
    # for i in indices:
    #     print(query_l[i], query_r[i], true_card[i])
    #     print("=====================================")
    # data = {'lb': query_l[indices].flatten(), 'ub': query_r[indices].flatten(), 'true_card': true_card[indices]}
    # df = pd.DataFrame(data)
    # print(df)
    # _df, dfs = generate_dataset()
    # print(_df.head())
    # print(_df.shape)
    # print(dfs.head())
    # print(dfs.shape)
