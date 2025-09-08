import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from colse.dataset_names import DatasetNames
from colse.datasets.params import ROW_PREFIX
from loguru import logger

from colse.data_path import get_data_path
from colse.df_utils import load_dataframe

TABLE_COLS = {
                "cast_info": ["movie_id","role_id"],
                "movie_companies": ["movie_id","company_id","company_type_id"],
                "movie_info_idx": ["movie_id","info_type_id"],
                "movie_keyword": ["movie_id","keyword_id"],
                "title": ["id","kind_id","production_year"],
                "movie_info": ["movie_id","info_type_id"],
            }

NO_OF_COLS = {
    "cast_info": 2,
    "movie_companies": 3,
    "movie_info_idx": 2,
    "movie_keyword": 2,
    "title": 3,
    "movie_info": 2,
}

def get_all_columns():
    # Query json order
    return [
        "cast_info:role_id",
        "movie_companies:company_id",
        "movie_companies:company_type_id",
        "movie_info:info_type_id",
        "movie_keyword:keyword_id",
        "title:kind_id",
        "title:production_year",
        "movie_info_idx:info_type_id",
    ]


def get_queries_imdb(**kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:

    dataset_type = DatasetNames.IMDB_DATA
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
                ub_list.append(equal_value)
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
