import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.df_utils import load_dataframe

TABLE_COLS = {
    "customers": ["id", "name", "city"],
    "orders": ["customer_id", "order_date", "amount"],
    "products": ["customer_id", "product_name", "quantity", "price"],
}

NO_OF_COLS = {
    "customers": 3,
    "orders": 3,
    "products": 4,
}


def get_all_columns():
    # Query json order
    return [
        "customers:name",
        "customers:city",
        "orders:order_date",
        "orders:amount",
        "products:product_name",
        "products:quantity",
        "products:price",
    ]


def generate_dataset(**kwargs):
    dataset_type = DatasetNames.CUSTOM_JOIN_DATA
    dataset_path = get_data_path(dataset_type) / "custom_join_dataset.xlsx"
    for index, table in enumerate(TABLE_COLS.keys()):
        df = pd.read_excel(
            dataset_path, sheet_name=f"table_{index + 1}", engine="openpyxl"
        )
        # df.columns = [f"{table}:{col}" for col in df.columns]
        df.to_parquet(dataset_path.parent / f"{table}.parquet")
        print(f"Saved {table}.parquet")


def get_queries_custom_join(
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dataset_type = DatasetNames.CUSTOM_JOIN_DATA
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
    entries = json.load(query_json.open())

    query_l = []
    query_r = []
    query_joined_tables = []
    for entry in entries[data_split]:
        query = entry[0]
        lb_list = []
        ub_list = []
        for key in query.keys():
            if isinstance(query[key], list) and query[key][0] == "[]":
                lb_list.append(query[key][1][0])
                ub_list.append(query[key][1][1])
            elif isinstance(query[key], list) and query[key][0] in ["<=", "<"]:
                lb_list.append(-np.inf)
                ub_list.append(query[key][1])
            elif isinstance(query[key], list) and query[key][0] in [">=", ">"]:
                lb_list.append(query[key][1])
                ub_list.append(np.inf)
            elif isinstance(query[key], list) and query[key][0] == "=":
                # Note - Here we are using a small range to approximate the equal condition [for IMDB dataset]
                equal_value = query[key][1]
                lb_list.append(equal_value)
                ub_list.append(equal_value + 1)
            else:
                lb_list.append(-np.inf)
                ub_list.append(np.inf)

        query_l.append(np.array(lb_list))
        query_r.append(np.array(ub_list))
        query_joined_tables.append(entry[1])

    if no_of_queries is not None:
        query_l = np.array(query_l[:no_of_queries]).astype(np.float64)
        query_r = np.array(query_r[:no_of_queries]).astype(np.float64)
        true_card = true_card[:no_of_queries].astype(np.float64)
        query_joined_tables = query_joined_tables[:no_of_queries]
    else:
        """convert all the data into float64"""
        query_l = np.array(query_l).astype(np.float64)
        query_r = np.array(query_r).astype(np.float64)
        true_card = true_card.astype(np.float64)
        query_joined_tables = query_joined_tables

    return query_l, query_r, true_card, query_joined_tables


if __name__ == "__main__":
    generate_dataset()
