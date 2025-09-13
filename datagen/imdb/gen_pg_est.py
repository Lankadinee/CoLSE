import re
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from loguru import logger
from tqdm import tqdm

RUN_ESTIMATES = True
RETRY_CONNECTION_WHEN_FAILED = False
EXECUTE_ANALYZE = False


current_dir = Path(__file__).resolve().parent


def count_queried_columns(sql):
    # Extract column names using a regex pattern that matches "col_something"
    # This assumes column names are like col_2, col_4, etc.
    columns = re.findall(r"\b(col_\w+)\b", sql)
    unique_columns = set(columns)

    return len(unique_columns), unique_columns


def create_connection(database_name, cardest_filename=False, query_no=0):
    logger.info(f"Connecting to {database_name}...")
    conn = psycopg2.connect(
        database=database_name,
        host="localhost",
        port=5432,
        password="dinee123",
        user="postgres",
    )
    conn.autocommit = True
    cursor = conn.cursor()
    time.sleep(2)
    return conn, cursor


def run_one_file(query_filename: str, labels_filename: str):
    conn, cursor = create_connection("imdb")

    labels_df = pd.read_csv(labels_filename)
    labels = labels_df["cardinality"].to_numpy().astype(int)

    imdb_sql_file = open(query_filename)
    queries = imdb_sql_file.readlines()
    imdb_sql_file.close()

    time.sleep(1)
    dict_list = []
    loop = tqdm(
        enumerate(zip(queries, labels)), total=len(queries), leave=True, file=sys.stdout
    )
    for no, (query, label) in loop:
        modified_query = query.replace("COUNT(*)", "*")
        sql_txt = "EXPLAIN (FORMAT JSON) " + modified_query.split("\n")[0]
        cursor.execute(sql_txt)
        res = cursor.fetchall()
        res_json = res[0][0][0]
        plan_rows = res_json["Plan"]["Plan Rows"]
        dict_list.append(
            {
                "query": query,
                "label": label,
                "plan_rows": plan_rows,
            }
        )

    df = pd.DataFrame(dict_list)
    logger.info(f"df: {df.head(70)}")

    cursor.close()
    conn.close()

    # Show stats
    total_queries = len(queries)
    logger.info(f"Total number of queries: {total_queries}")
    logger.info(f"Total number of queries processed: {len(df)}")

    """Write to a csv file"""
    text = ""
    for row in df.itertuples():
        text += f"{row.query.strip()}||{row.label}||{row.plan_rows}\n"

    new = ["imdb_mii", "imdb_ci", "imdb_t", "imdb_mi", "imdb_mk", "imdb_mc"]
    old = ["mi_idx", "ci", "t", "mi", "mk", "mc"]
    for n, o in zip(new, old):
        for p in [",", ".", " WHERE"]:
            text = text.replace(o + p, n + p)

    with open("/datadrive500/PRICE/datas/workloads/test/imdb/workloads.sql", "w") as f:
        f.write(text)


if __name__ == "__main__":
    query_filename = "/datadrive500/CoLSE/data/imdb/job-light.sql"
    labels_filename = "/datadrive500/CoLSE/data/imdb/label_train.csv"
    run_one_file(query_filename, labels_filename)
