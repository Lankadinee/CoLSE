import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

from colse.datasets.dataset_forest import generate_dataset


current_dir = Path(__file__).parent
dataset_dir = current_dir.joinpath("../../data/forest")

def actual_cdf(df, lb, ub=None):
        X = df.to_numpy().transpose()
        value = 1
        index = 0

        if ub is None:
            for x1 in lb:
                value *= X[index] <= x1
                index += 1
        else:
            for x1, x2 in zip(lb, ub):
                value *= (X[index] >= x1) * (X[index] <= x2)
                index += 1

        return value.sum()

def get_actual_cardinality(df, query_l, query_r, verbose=False):
        logger.info("Generating true cardinality") if verbose else None
        actual_card = []
        for ub, lb in tqdm(zip(query_r, query_l), total=query_l.shape[0]):
            actual_cdf_value = actual_cdf(df, lb, ub)
            actual_card.append(actual_cdf_value if actual_cdf_value > 1 else 1)
        return np.array(actual_card)

def load_queries(query_type='train'):
    """Load queries"""
    query_json = dataset_dir.joinpath("query.json")
    logger.info(f"Loading queries from {query_json.absolute()}")
    queries = json.load(query_json.open())
    # training_queries = queries['train']
    # validation_queries = queries['valid']
    # testing_queries = queries['test']

    query_l = []
    query_r = []
    for query in queries[query_type]:
        query = query[0]
        lb_list = []
        ub_list = []
        for key in query.keys():
            if isinstance(query[key], list) and query[key][0] == '[]':
                lb_list.append(query[key][1][0])
                ub_list.append(query[key][1][1])
            elif isinstance(query[key], list) and query[key][0] == '<=':
                lb_list.append(-np.inf)
                ub_list.append(query[key][1])
            elif isinstance(query[key], list) and query[key][0] == '>=':
                lb_list.append(query[key][1])
                ub_list.append(np.inf)
            else:
                lb_list.append(-np.inf)
                ub_list.append(np.inf)

        query_l.append(np.array(lb_list))
        query_r.append(np.array(ub_list))

    """Load the dataframe"""
    df = generate_dataset(no_of_rows=None)

    return df, np.array(query_l), np.array(query_r)


if __name__ == '__main__':
    df, query_l, query_r = load_queries(query_type='test')
    true_card = get_actual_cardinality(df, query_l, query_r, verbose=True)
    print(true_card)
    selectivity = true_card / df.shape[0]
    df_card = pd.DataFrame({
    'cardinality': true_card,
    'selectivity': selectivity
    })
    print(df_card.head())
    df_card.to_csv(dataset_dir.joinpath("label_test.csv"), index=False)