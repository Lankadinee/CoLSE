import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from colse_enums import get_common_database_name
from get_list_of_files import get_all_input_files
from loguru import logger
from rich.console import Console
from rich.table import Table
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


def main(dataset: str, container_name):
    for cardest_filename in get_all_input_files(
        container_name, "/var/lib/pgsql/13.1/data/"
    ):
        run_one_file(dataset, cardest_filename)


def run_one_file(database_name: str, cardest_filename: str):
    logger.info("------------------------------------------------------")
    database_common_name = get_common_database_name(database_name)
    database_split_name = database_name.split("_")[0]
    logger.info(
        f"Running {cardest_filename} for database {database_name} with common name {database_common_name}"
    )
    sql_file = f"./workloads/{database_name}/{database_common_name}.sql"

    conn, cursor = create_connection(database_split_name)

    export_dirpath = current_dir / f"../plan_cost/{database_name}/"
    export_filepath = (
        export_dirpath / f"{cardest_filename.split('.')[0] + '_cost.xlsx'}"
    )
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
        logger.info(f"Using estimates from {cardest_filename}")
        # Single table queries
        # cursor.execute('SET logger.info_single_tbl_queries=true')
        # cursor.execute("SET enable_indexscan=on;")
        logger.info("Setting ml_cardest_enabled=true;")
        cursor.execute("SET ml_cardest_enabled=true;")
        logger.info(f"Setting ml_cardest_fname='{cardest_filename}';")
        cursor.execute(f"SET ml_cardest_fname='{cardest_filename}';")
        logger.info("Setting query_no=0;")
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
    loop = tqdm(enumerate(queries), total=len(queries), leave=True, file=sys.stdout)
    for no, query in loop:
        scan_type = []
        # sql_txt = "EXPLAIN (FORMAT JSON)SELECT COUNT(*) FROM forest;"
        # EXPLAIN (FORMAT JSON)SELECT COUNT(*) FROM badges as b, users as u WHERE b.UserId= u.Id AND u.UpVotes>=0;
        if EXECUTE_ANALYZE:
            sql_txt = "EXPLAIN (ANALYZE, FORMAT JSON) " + query.split("\n")[0]
        else:
            sql_txt = "EXPLAIN (FORMAT JSON) " + query.split("\n")[0]
        # cursor.execute(sql_txt)
        # res = cursor.fetchall()
        # logger.info(f"Executing {no}-th query: {sql_txt}")
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
        res_json = res[0][0][0]

        if EXECUTE_ANALYZE:
            query_execution_time = res_json["Execution Time"]
            query_planning_time = res_json["Planning Time"]
            query_total_time = query_execution_time + query_planning_time
            dict_list.append(
                {
                    "index": no,
                    "query_total_time": query_total_time,
                    "query_execution_time": query_execution_time,
                    "query_planning_time": query_planning_time,
                    "query": query.split("\n")[0],
                }
            )
        else:
            total_cost = res_json["Plan"]["Total Cost"]
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
                        "query": query.split("\n")[0],
                    }
                )
            else:
                dict_list.append(
                    {"total_cost_true": total_cost, "access_path": scan_type}
                )

    logger.info("Used estimates from ", cardest_filename)
    df = pd.DataFrame(dict_list)
    logger.info(f"df: {df.head()}")

    cursor.close()
    conn.close()

    # Show stats
    total_queries = len(queries)
    logger.info(f"Total number of queries: {total_queries}")
    logger.info(f"Total number of queries processed: {len(df)}")

    if EXECUTE_ANALYZE:
        # show max, min and average of query_total_time
        logger.info(f"Max query total time: {df['query_total_time'].max()}")
        logger.info(f"Min query total time: {df['query_total_time'].min()}")
        logger.info(f"Average query total time: {df['query_total_time'].mean()}")

        export_filepath = (
            export_dirpath / f"{cardest_filename.split('.')[0] + '_cost_analyze.xlsx'}"
        )

    """Write to a csv file"""
    logger.info(f"export_filepath: {export_filepath}")
    df.to_excel(export_filepath.as_posix(), index=False)

    # Show unique access paths
    if EXECUTE_ANALYZE:
        return

    unique_access_paths_counts = df["access_path"].value_counts()

    console = Console()
    table = Table(title="Access Path Counts")
    table.add_column("Table Scan Type", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    for path, count in unique_access_paths_counts.items():
        table.add_row(",".join(path), str(count), f"{count / total_queries * 100:.2f}%")

    console.print(table)


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
        help="The dataset to use (power or forest).",
    )
    parser.add_argument(
        "--container_name", help="Name of the container to run the queries."
    )
    parser.add_argument("--filename", default="NA", help="Cardest filename")

    args = parser.parse_args()
    database_name = args.database_name
    if args.filename == "NA":
        main(database_name, args.container_name)
    else:
        run_one_file(database_name, args.filename)
