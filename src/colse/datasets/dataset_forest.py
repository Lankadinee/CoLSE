import json
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from colse.datasets.params import ROW_PREFIX
from loguru import logger
from sklearn.model_selection import train_test_split

from colse.data_path import get_data_path
from colse.df_utils import load_dataframe

current_dir = Path(__file__).parent
dataset_dir = dataset_dir = get_data_path() / "forest/"


def generate_dataset(**kwargs):
    nrows = kwargs.get("no_of_rows", 500_000)
    no_of_columns = kwargs.get("no_of_cols", None)
    selected_cols = kwargs.get("selected_cols", None)

    logger.info("Loading forest dataset...")
    df = load_dataframe(dataset_dir / "forest.csv")
    # df = pd.read_excel(dataset_dir.joinpath("forest_deq_rounded.xlsx"))
    logger.info("Forest dataframe loaded.")

    nrows = df.shape[0] if nrows is None else nrows

    logger.info(f"Convert the first {nrows} rows to numpy array")
    attr1 = df["Elevation"].to_numpy()[:nrows]
    attr2 = df["Aspect"].to_numpy()[:nrows]
    attr3 = df["Slope"].to_numpy()[:nrows]
    attr4 = df["Horizontal_Distance_To_Hydrology"].to_numpy()[:nrows]
    attr5 = df["Vertical_Distance_To_Hydrology"].to_numpy()[:nrows]
    attr6 = df["Horizontal_Distance_To_Roadways"].to_numpy()[:nrows]
    attr7 = df["Hillshade_9am"].to_numpy()[:nrows]
    attr8 = df["Hillshade_Noon"].to_numpy()[:nrows]
    attr9 = df["Hillshade_3pm"].to_numpy()[:nrows]
    attr10 = df["Horizontal_Distance_To_Fire_Points"].to_numpy()[:nrows]
    # data = df.to_numpy().astype(int)

    logger.info("Stacking the attributes into a 2D array")
    # Stack the attributes into a 2D array
    data = np.column_stack((attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10))

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 11)])

    if no_of_columns:
        df = df.iloc[:, :no_of_columns]

    if selected_cols:
        df = df.iloc[:, selected_cols]

    # df = df.astype(np.float64)
    return df


def get_queries_forest(**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    data_split = kwargs.get("data_split", "train")
    no_of_queries = kwargs.get("no_of_queries", None)
    min_value = kwargs.get("min_value", 1)
    type_test = kwargs.get("type_test", False)
    is_test_set = kwargs.get("is_test_set", False)

    """Load queries"""
    query_json = dataset_dir / "query.json"
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
            else:
                lb_list.append(-np.inf)
                ub_list.append(np.inf)

        query_l.append(np.array(lb_list))
        query_r.append(np.array(ub_list))

    "Load true cardinality"
    labels = load_dataframe(dataset_dir / f"label_{data_split}.csv")
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
        query_l = np.array(query_l[:no_of_queries]).astype(np.float64)
        query_r = np.array(query_r[:no_of_queries]).astype(np.float64)
        true_card = true_card[:no_of_queries].astype(np.float64)
    else:
        """convert all the data into float64"""
        query_l = np.array(query_l).astype(np.float64)
        query_r = np.array(query_r).astype(np.float64)
        true_card = true_card.astype(np.float64)

    if is_test_set:
        index_list = np.arange(len(query_l))
        training_indices, test_indices = train_test_split(index_list, test_size=0.2, random_state=42)
        if type_test:
            return query_l[test_indices], query_r[test_indices], true_card[test_indices]
        else:
            return query_l[training_indices], query_r[training_indices], true_card[training_indices]
    else:
        return query_l, query_r, true_card


def queried_values(query_list):
    """Get the queried values"""
    query_dict = {}
    for i in range(query_list.shape[0]):
        for j in range(query_list[i].shape[0]):
            if query_list[i][j] != -np.inf:
                query_dict[j] = query_dict.get(j, []) + [query_list[i][j]]
    return query_dict


def get_sample_queries(no_of_queries, queried_columns: list = None, no_inf=True):
    query_l, query_r, true_card = get_queries_forest()
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
            col_sample = np.array([np.random.choice(unique_values_dict[col_index], no_of_queries) for _ in range(2)])
            col_sample_sorted = np.sort(col_sample, axis=0)
            ub_queries, lb_queries = col_sample_sorted[1, :], col_sample_sorted[0, :]
        else:
            if no_inf:
                continue
            ub_queries, lb_queries = np.array([+np.inf] * no_of_queries), np.array([-np.inf] * no_of_queries)
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
    query_l, query_r, true_card = get_queries_forest()
    query_len = []
    for lb, ub in zip(query_l, query_r):
        count = 0
        for i in range(len(lb)):
            if not (np.any(lb[i] == -np.inf) and np.any(ub[i] == np.inf)):
                count = count + 1
        query_len.append(count)

    count_dict = Counter(query_len)
    total_points = sum(count_dict.values())

    print(f"Total points: {total_points}")
    print(count_dict)

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
