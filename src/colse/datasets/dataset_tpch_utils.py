import json
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.params import ROW_PREFIX
from colse.df_utils import load_dataframe, save_dataframe


def tpch_lineitem_preprocess(dataset_type: DatasetNames, skip_if_exists: bool = False):

    input_file_path = get_data_path(dataset_type) / "original.parquet"
    output_file_path = dataset_type.get_file_path(exist_check=False)

    if skip_if_exists and output_file_path.exists():
        logger.info(f"Skipping {dataset_type} dataset preprocessing because it already exists")
        return True

    logger.info(f"Preprocessing {dataset_type} dataset from - {input_file_path}")
    df = load_dataframe(input_file_path)

    df["l_commitdate"] = pd.to_datetime(df["l_commitdate"], format="%Y-%m-%d")
    df["l_receiptdate"] = pd.to_datetime(df["l_receiptdate"], format="%Y-%m-%d")
    df["l_shipdate"] = pd.to_datetime(df["l_shipdate"], format="%Y-%m-%d")

    """find minimum date and subtract from all dates"""
    min_date = min(
        df["l_commitdate"].min(), df["l_receiptdate"].min(), df["l_shipdate"].min()
    )
    df["l_commitdate"] = (df["l_commitdate"] - min_date).dt.days
    df["l_receiptdate"] = (df["l_receiptdate"] - min_date).dt.days
    df["l_shipdate"] = (df["l_shipdate"] - min_date).dt.days

    save_dataframe(df, output_file_path)
    logger.info(f"Preprocessed {dataset_type} dataset saved to - {output_file_path}")
    return True


def generate_dataset_tpch_lineitem(**kwargs):
    dataset_type = kwargs.get("dataset_type", None)
    if dataset_type is None:
        raise ValueError("dataset_type is required")
    nrows = kwargs.get("no_of_rows", 500_000)
    no_of_columns = kwargs.get("no_of_cols", None)
    selected_cols = kwargs.get("selected_cols", None)
    data_file_name = kwargs.get("data_file_name", None)
    assert data_file_name is not None, "data_file_name is required"

    dataset_path = get_data_path(dataset_type) / data_file_name
    logger.info(f"Loading {dataset_type} dataset from: {dataset_path}")
    df = load_dataframe(dataset_path)

    nrows = df.shape[0] if nrows is None else nrows

    attr1 = df["l_orderkey"].to_numpy()[:nrows]
    attr2 = df["l_partkey"].to_numpy()[:nrows]
    attr3 = df["l_suppkey"].to_numpy()[:nrows]
    attr4 = df["l_linenumber"].to_numpy()[:nrows]
    attr5 = df["l_quantity"].to_numpy()[:nrows]
    attr6 = df["l_extendedprice"].to_numpy()[:nrows]
    attr7 = df["l_discount"].to_numpy()[:nrows]
    attr8 = df["l_tax"].to_numpy()[:nrows]
    attr9 = df["l_returnflag"].to_numpy()[:nrows]
    attr10 = df["l_linestatus"].to_numpy()[:nrows]
    attr11 = df["l_shipdate"].to_numpy()[:nrows]
    attr12 = df["l_commitdate"].to_numpy()[:nrows]
    attr13 = df["l_receiptdate"].to_numpy()[:nrows]
    attr14 = df["l_shipinstruct"].to_numpy()[:nrows]
    attr15 = df["l_shipmode"].to_numpy()[:nrows]
    # data = df.to_numpy().astype(int)

    # Stack the attributes into a 2D array
    data = np.column_stack(
        (
            attr1,
            attr2,
            attr3,
            attr4,
            attr5,
            attr6,
            attr7,
            attr8,
            attr9,
            attr10,
            attr11,
            attr12,
            attr13,
            attr14,
            attr15,
        )
    )

    new_df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 16)])

    for new_col, old_col in zip(new_df.columns, df.columns):
        new_df[new_col] = new_df[new_col].astype(df[old_col].dtype)

    if no_of_columns:
        new_df = new_df.iloc[:, :no_of_columns]

    if selected_cols:
        new_df = new_df.iloc[:, selected_cols]

    # df = df.astype(np.float64)
    return new_df


def get_queries_tpch_lineitem(
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    MIN_DATE = "1992-01-02"

    dataset_type = kwargs.get("dataset_type", None)
    if dataset_type is None:
        raise ValueError("dataset_type is required")
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
    label_file_name = f"label_{dataset_type}_{data_split}.csv"
    logger.info(f"Loading true cardinality from {dataset_dir / label_file_name}")
    labels = load_dataframe(dataset_dir / label_file_name)
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
                if (
                    key == "l_shipdate"
                    or key == "l_commitdate"
                    or key == "l_receiptdate"
                ):
                    no_of_days = (
                        datetime.strptime(query[key][1], "%Y-%m-%d")
                        - datetime.strptime(MIN_DATE, "%Y-%m-%d")
                    ).days
                    lb_list.append(no_of_days)
                    ub_list.append(no_of_days)
                else:
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
