import json
from loguru import logger

import pandas as pd

from colse.df_utils import load_dataframe, save_dataframe

def json_to_sql(query, dataset_name):
    query = query[0]
    sql_query = f"SELECT * FROM {dataset_name} WHERE "

    for key in query.keys():
        if isinstance(query[key], list) and query[key][0] == "[]":
            sql_query += f"{key} >= {query[key][1][0]} AND {key} <= {query[key][1][1]} AND "
        elif isinstance(query[key], list) and query[key][0] == "<=":
            sql_query += f"{key} <= {query[key][1]} AND "
        elif isinstance(query[key], list) and query[key][0] == ">=":
            sql_query += f"{key} >= {query[key][1]} AND "
        elif isinstance(query[key], list) and query[key][0] == "=":
            sql_query += f"{key} = '{query[key][1]}' AND "

    """Remove the last 'AND'"""
    sql_query = sql_query[:-4]
    """Remove last space and add a semicolon"""
    sql_query = sql_query.strip() + ";"
    return sql_query

def json_file_to_sql_file(json_file_path, sql_file_path, dataset_name):
    queries = json.load(open(json_file_path))
    converted_query_list = []
    for query in queries['test']:
        converted_query = json_to_sql(query, dataset_name)
        converted_query_list.append(converted_query)

    with open(sql_file_path, "w") as f:
        for query in converted_query_list:
            f.write(query)
            f.write("\n")
    print("Saved to ", sql_file_path)
    return True

def write_true_card_to_txt(label_file_path, dataset_name, update_type):
    labels = pd.read_csv(label_file_path)
    true_card = labels["cardinality"].to_numpy().astype(int)

    txt_file = f"workloads/{dataset_name}_{update_type}/estimates/true_card.txt"
    with open(txt_file, "w") as f:
        for tc in true_card:
            f.write(str(tc))
            f.write("\n")
    print("Saved to ", txt_file)


def excel_to_estimates_csv(excel_path, estimates_path, no_of_rows):
    logger.info("Excel to estimates csv")
    # this is our excel files
    df = load_dataframe(excel_path)

    new_df = pd.DataFrame(columns=["cardinality", "selectivity"])
    new_df["cardinality"] = df["y_bar_2"] * no_of_rows 
    new_df["selectivity"] = df["y_bar_2"]
    save_dataframe(new_df, estimates_path)
    logger.info(f"Saved to {estimates_path}")
    return True

def csv_to_estimates_csv(csv_path, estimates_path, no_of_rows):
    logger.info("CSV to estimates csv")
    # this is ACE's csv files
    df = load_dataframe(csv_path)
    new_df = pd.DataFrame(columns=["cardinality", "selectivity"])
    new_df["cardinality"] = df["predict"] 
    new_df["selectivity"] = df["predict"] / no_of_rows
    save_dataframe(new_df, estimates_path)
    logger.info(f"Saved to {estimates_path}")
    return True


def test():
    dataset_name = "census"
    update_type= "skew_0.2"
    json_file = f"data/{dataset_name}/data_updates/query_{update_type}.json"
    sql_file = f"workloads/{dataset_name}_{update_type}/{dataset_name}_{update_type}.sql"
    json_file_to_sql_file(json_file, sql_file)
    """Write true cardinality to a txt file"""
    label_file_path = f"data/{dataset_name}/data_updates/label_{update_type}_test.csv"
    write_true_card_to_txt(label_file_path, dataset_name, update_type)
