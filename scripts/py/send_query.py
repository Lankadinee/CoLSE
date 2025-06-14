import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from get_list_of_files import get_all_unprocessed_txt_files
from loguru import logger
from tqdm import tqdm

from colse_enums import Datasets

RUN_ESTIMATES = True
RETRY_CONNECTION_WHEN_FAILED = False

METADATA = {
    "power": {
        "database_name": "power",
        "sql_file": "./workloads/power/power_query.sql",
    },
    "forest": {
        "database_name": "forest",
        "sql_file": "./workloads/forest/forest_query.sql",
    },
    "census": {
        "database_name": "census",
        "sql_file": "./workloads/census/census_query.sql",
    },
    "dmv": {
        "database_name": "dmv",
        "sql_file": "./workloads/dmv/dmv_query.sql",
    },
    "tpch_sf2_z0": {
        "database_name": "tpch_sf2_z0",
        "sql_file": "./workloads/tpch_sf2_z0/tpch_sf2_z0.sql",
    },
    "tpch_sf2_z1": {
        "database_name": "tpch_sf2_z1",
        "sql_file": "./workloads/tpch_sf2_z1/tpch_sf2_z1.sql",
    },
    "tpch_sf2_z2": {
        "database_name": "tpch_sf2_z2",
        "sql_file": "./workloads/tpch_sf2_z2/tpch_sf2_z2.sql",
    },
    "tpch_sf2_z3": {
        "database_name": "tpch_sf2_z3",
        "sql_file": "./workloads/tpch_sf2_z3/tpch_sf2_z3.sql",
    },
    "tpch_sf2_z4": {
        "database_name": "tpch_sf2_z4",
        "sql_file": "./workloads/tpch_sf2_z4/tpch_sf2_z4.sql",
    },
    "tpch_10": {
        "database_name": "tpch_10",
        "sql_file": "./workloads/tpch_10/tpch_10.sql",
    },
    "tpch_20": {
        "database_name": "tpch_20",
        "sql_file": "./workloads/tpch_20/tpch_20.sql",
    },
    "synthetic_correlated_2": {
        "database_name": "synthetic_correlated_2",
        "sql_file": "./workloads/synthetic_correlated_2/synthetic_correlated_2.sql",
    },
    "synthetic_correlated_3": {
        "database_name": "synthetic_correlated_3",
        "sql_file": "./workloads/synthetic_correlated_3/synthetic_correlated_3.sql",
    },
    "synthetic_correlated_4": {
        "database_name": "synthetic_correlated_4",
        "sql_file": "./workloads/synthetic_correlated_4/synthetic_correlated_4.sql",
    },
    "synthetic_correlated_6": {
        "database_name": "synthetic_correlated_6",
        "sql_file": "./workloads/synthetic_correlated_6/synthetic_correlated_6.sql",
    },
    "synthetic_correlated_8": {
        "database_name": "synthetic_correlated_8",
        "sql_file": "./workloads/synthetic_correlated_8/synthetic_correlated_8.sql",
    },
    "synthetic_correlated_10": {
        "database_name": "synthetic_correlated_10",
        "sql_file": "./workloads/synthetic_correlated_10/synthetic_correlated_10.sql",
    },
    "tpch_lineitem_10": {
        "database_name": "tpch_lineitem_10",
        "sql_file": "./workloads/tpch_lineitem_10/tpch_lineitem_10.sql",
    },
    "tpch_lineitem_20": {
        "database_name": "tpch_lineitem_20",
        "sql_file": "./workloads/tpch_lineitem_20/tpch_lineitem_20.sql",
    },
    "correlated_02": {
        "database_name": "correlated_02",
        "sql_file": "./workloads/correlated_02/correlated_02.sql",
    },
    "correlated_04": {
        "database_name": "correlated_04",
        "sql_file": "./workloads/correlated_04/correlated_04.sql",
    },
    "correlated_06": {
        "database_name": "correlated_06",
        "sql_file": "./workloads/correlated_06/correlated_06.sql",
    },
    "correlated_08": {
        "database_name": "correlated_08",
        "sql_file": "./workloads/correlated_08/correlated_08.sql",
    },
    "correlated_1_2": {
        "database_name": "correlated_1_2",
        "sql_file": "./workloads/correlated_1_2/correlated_1_2.sql",
    },
    "random": {
        "database_name": "random",
        "sql_file": "./workloads/random/random_query.sql",
    },
    "census_cor_02": {
        "database_name": "census_cor_02",
        "sql_file": "./workloads/census_cor_02/census_cor_02.sql",
    },
    "census_ind_02": {
        "database_name": "census_ind_02",
        "sql_file": "./workloads/census_ind_02/census_ind_02.sql",
    },
    "census_skew_02": {
        "database_name": "census_skew_02",
        "sql_file": "./workloads/census_skew_02/census_skew_02.sql",
    },
}

current_dir = Path(__file__).resolve().parent

def count_queried_columns(sql):
    # Extract column names using a regex pattern that matches "col_something"
    # This assumes column names are like col_2, col_4, etc.
    columns = re.findall(r'\b(col_\w+)\b', sql)
    unique_columns = set(columns)
    
    return len(unique_columns), unique_columns
    
def create_connection(database_name, cardest_filename=False, query_no=0):
    logger.info(f"Connecting to {database_name}...")
    conn = psycopg2.connect(
        database=database_name,
        host="localhost",
        port=5430,
        password="postgres",
        user="postgres",
    )
    conn.autocommit = True
    cursor = conn.cursor()

    if cardest_filename:
        logger.info(f"Using {cardest_filename} for estimates. Setting up the config...")
        # cursor.execute("SET debug_card_est=true;")
        # cursor.execute("SET logger.info_sub_queries=true;")

        if RUN_ESTIMATES:
            # Single table queries
            # cursor.execute('SET logger.info_single_tbl_queries=true')
            cursor.execute("SET ml_cardest_enabled=true;")
            cursor.execute(f"SET ml_cardest_fname='{cardest_filename}';")
            cursor.execute(f"SET query_no={query_no};")

    time.sleep(2)
    return conn, cursor


def main(dataset, container_name):
    for cardest_filename in get_all_unprocessed_txt_files(
        container_name, "/var/lib/pgsql/13.1/data/"
    ):
        run_one_file(dataset, cardest_filename)


def run_one_file(dataset: Datasets, cardest_filename: str):
    database_name = dataset.value
    sql_file = dataset.get_sql_file_path()

    conn, cursor = create_connection(database_name)
    logger.info(f"Processing {cardest_filename}")

    export_dirpath = current_dir / f"../plan_cost/{database_name}/"
    export_filepath = export_dirpath / f'{cardest_filename.split(".")[0] + "_cost.csv"}'
    logger.info(f"Exporting to {export_filepath}")
    if not export_dirpath.exists():
        export_dirpath.mkdir(parents=True)

    if export_filepath.exists():
        logger.info(f"File {export_filepath} already exists. Deleting...")
        export_filepath.unlink()

        # imdb_sql_file = open("/home/titan/phd/megadrive/End-to-End-CardEst-Benchmark/workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql")
    imdb_sql_file = open(sql_file)
    queries = imdb_sql_file.readlines()
    imdb_sql_file.close()

    if os.path.exists("/Users/hanyuxing/pgsql/13.1/data/join_est_record_job.txt"):
        os.remove("/Users/hanyuxing/pgsql/13.1/data/join_est_record_job.txt")

    """ ReadMe
        stats=# SET ml_cardest_enabled=true; ## for single table
        stats=# SET ml_joinest_enabled=true; ## for multi-table
        stats=# SET query_no=0; ##for single table
        stats=# SET join_est_no=0; ##for multi-table
        stats=# SET ml_cardest_fname='stats_CEB_sub_queries_bayescard.txt'; ## for single table
        stats=# SET ml_joinest_fname='stats_CEB_sub_queries_bayescard.txt'; ## for multi-table
        """
    # cursor.execute("SET debug_card_est=true;")
    # cursor.execute("SET logger.info_sub_queries=true;")

    if RUN_ESTIMATES:
        logger.info("Using estimates from ", cardest_filename)
        # Single table queries
        # cursor.execute('SET logger.info_single_tbl_queries=true')
        # cursor.execute("SET enable_indexscan=on;")
        cursor.execute("SET ml_cardest_enabled=true;")
        cursor.execute(f"SET ml_cardest_fname='{cardest_filename}';")
        cursor.execute("SET query_no=0;")

        # Join queries
        # cursor.execute("SET ml_joinest_enabled=true;")
        # cursor.execute("SET join_est_no=0;")
        # cursor.execute("SET ml_joinest_fname='stats_CEB_join_queries_bayescard.txt';")
        # conn.commit()

        # conn = psycopg2.connect(database="stats", host="localhost", port=5431, password="postgres", user="postgres")
        # cursor = conn.cursor()

    time.sleep(1)
    dict_list = []
    loop = tqdm(
        enumerate(queries), total=len(queries), leave=True, file=sys.stdout
    )
    for no, query in loop:
        scan_type = []
        # sql_txt = "EXPLAIN (FORMAT JSON)SELECT COUNT(*) FROM forest;"
        # EXPLAIN (FORMAT JSON)SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId= u.Id AND u.UpVotes>=0;
        sql_txt = "EXPLAIN (FORMAT JSON) " + query.split("\n")[0]
        # cursor.execute(sql_txt)
        # res = cursor.fetchall()
        logger.info(f"Executing {no}-th query: {sql_txt}")
        retry_count = 0
        while True:
            try:
                if retry_count > 0:
                    conn, cursor = create_connection(
                        database_name, cardest_filename, query_no=no
                    )
                cursor.execute(sql_txt)
                no_cols = count_queried_columns(query.split("\n")[0])
                # loop.set_description(
                #     f"Executing {no}-th query: {sql_txt}"
                # )
                res = cursor.fetchall()
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.info(e)
                logger.error("Connection error")
                cursor.close()
                conn.close()
                retry_count += 1
                if retry_count > 5 or not RETRY_CONNECTION_WHEN_FAILED:
                    logger.info("Failed to execute the query.")
                    break
                logger.info("Retrying... ", retry_count)
                time.sleep(3)
                continue
        logger.info(res)
        res_json = res[0][0][0]
        total_cost = res_json["Plan"]["Total Cost"]
        # query_total_time = res_json["Execution Time"] + res_json["Planning Time"]
        scan_type.append(res_json["Plan"]["Node Type"])
        plans = res_json["Plan"].get("Plans", [])
        if plans:
            scan_type.append(plans[0]["Node Type"])
        rows = res_json["Plan"]["Plan Rows"]

        if RUN_ESTIMATES:
            dict_list.append(
                {
                    "index": no,
                    "total_cost_estimates": total_cost,
                    "access_path": scan_type,
                    "input_card_est": rows,
                    "no_queried_columns": no_cols[0],
                    # "query_total_time": query_total_time,
                    # "query_execution_time": res_json["Execution Time"],
                    # "query_planning_time": res_json["Planning Time"],
                    "query": query.split("\n")[0]
                }
            )
        else:
            dict_list.append(
                {"total_cost_true": total_cost, 
                 "access_path": scan_type}
            )

    logger.info("Used estimates from ", cardest_filename)
    df = pd.DataFrame(dict_list)

    if export_filepath.exists():
        df_file = pd.read_csv(export_filepath)
        df = pd.concat([df_file, df], axis=1)

    """Write to a csv file"""
    df.to_csv(export_filepath.as_posix(), index=False)

    cursor.close()
    conn.close()

    # Show stats
    logger.info("Total number of queries: ", len(queries))
    logger.info("Total number of queries processed: ", len(df))


    # Show unique access paths
    unique_access_paths_counts = df["access_path"].value_counts()

    logger.info("Table Scan type\tCount")
    for path, count in unique_access_paths_counts.items():
        logger.info(f"{path}\t: {count}")

def test():
    # Test the function with a sample dataset
    dataset = "correlated_04"
    cardest_filename = "mhist_30000_correlated_04_estimates.csv"
    run_one_file(dataset, cardest_filename)


if __name__ == "__main__":
    logger.info("Starting the process...")
    # test()
    parser = argparse.ArgumentParser(
        description="Execute SQL queries and export cost estimates."
    )
    parser.add_argument(
        "--database_name",
        choices=METADATA.keys(),
        help="The dataset to use (power or forest).",
    )
    parser.add_argument(
        "--container_name", help="Name of the container to run the queries."
    )
    parser.add_argument("--filename", default="NA", help="Cardest filename")

    args = parser.parse_args()
    dataset = Datasets(args.database_name)
    if args.filename == "NA":
        main(dataset, args.container_name)
    else:
        run_one_file(dataset, args.filename)
