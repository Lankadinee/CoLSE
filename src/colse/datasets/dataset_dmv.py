import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.params import ROW_PREFIX
from colse.df_utils import load_dataframe

def generate_dataset(**kwargs):
    dataset_type = DatasetNames.DMV_DATA
    nrows = kwargs.get("no_of_rows", 500_000)
    no_of_columns = kwargs.get("no_of_cols", None)
    selected_cols = kwargs.get("selected_cols", None)
    data_file_name = kwargs.get("data_file_name", None)
    assert data_file_name is not None, "data_file_name is required"

    dataset_path = get_data_path(dataset_type) / data_file_name
    logger.info(f"Loading {dataset_type} dataset from: {dataset_path}")
    df = load_dataframe(dataset_path)
    logger.info("DMV dataframe loaded.")

    nrows = df.shape[0] if nrows is None else nrows
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
    dataset_type = DatasetNames.DMV_DATA
    data_split = kwargs.get("data_split", "train")
    no_of_queries = kwargs.get("no_of_queries", None)

    """Load queries"""
    dataset_dir = get_data_path(dataset_type)
    query_file_name = kwargs.get("query_file_name", None)

    """Load queries"""
    if query_file_name is None:
        query_json = dataset_dir / "query.json"
    else:
        query_json = dataset_dir / query_file_name

    if not query_json.exists():
        raise FileNotFoundError(f"File {query_json.absolute()} not found")

    "Load true cardinality"
    label_file_name = f"{query_json.parent}/{query_json.stem.replace('query', 'label')}_{data_split}.csv"
    label_file_path = dataset_dir / label_file_name
    logger.info(f"Loading true cardinality from {label_file_path}")
    labels = load_dataframe(label_file_path)
    true_card = labels["cardinality"].to_numpy().astype(int)

    logger.info(f"Loading queries from {query_json.absolute()}")
    queries = json.load(query_json.open())

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

    if no_of_queries is not None:
        query_l = np.array(query_l[:no_of_queries])
        query_r = np.array(query_r[:no_of_queries])
        true_card = true_card[:no_of_queries]
    else:
        """convert all the data into float64"""
        query_l = np.array(query_l)
        query_r = np.array(query_r)
        true_card = true_card

    return query_l, query_r, true_card

