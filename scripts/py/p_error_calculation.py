import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def calculate_p_error(estimates_cost_file_path, true_cost_file_path):
    df_estimates = pd.read_csv(estimates_cost_file_path)
    df_true = pd.read_csv(true_cost_file_path)
    
    cost_estimates = df_estimates["total_cost_estimates"].values
    cost_true = df_true["total_cost_estimates"].values

    p_errors = np.maximum(cost_estimates, cost_true) / np.minimum(cost_estimates, cost_true)

    access_path_estimates = df_estimates["access_path"].values
    access_path_true = df_true["access_path"].values

    """Check whether the access paths are the same and get a count of the same access paths"""
    same_access_paths = 0
    sub_optimal_access_path_id_list = []
    for i in range(len(access_path_estimates)):
        if access_path_estimates[i] == access_path_true[i]:
            same_access_paths += 1
        else:
            sub_optimal_access_path_id_list.append(i)
    
    # query_execution_time_estimates = df_estimates["query_execution_time"].values
    # query_execution_time_true = df_true["query_execution_time"].values

    # high_execution_time = []
    # for i in range(len(query_execution_time_estimates)):
    #     if query_execution_time_estimates[i] > query_execution_time_true[i]:
    #         high_execution_time.append(1)
    #     else:
    #         high_execution_time.append(0)

    # df["p_error"] = df.apply(
    #     lambda row: max(row["total_cost_estimates"], row["total_cost_true"]) / 
    #     (min(row["total_cost_estimates"], row["total_cost_true"])), axis=1)
    
    """Calculate the percentiles of p_error"""
    percentiles_values = [50, 90, 95, 99, 100]
    
    result_file_path = Path(estimates_cost_file_path).with_suffix(".txt")
    with open(result_file_path, "a") as f:
        f.write(f"Estimates cost file: {estimates_cost_file_path}\n")
        f.write(f"True cost file: {true_cost_file_path}\n")
        f.write(f"Total number of queries: {len(p_errors)}\n")
        f.write(f"Total number of same access paths: {same_access_paths}\n")
        # f.write(f"Total number of queries with high execution time: {sum(high_execution_time)}\n")
        # f.write(f"Mean query execution time: {np.mean(query_execution_time_estimates)}\n")
        # f.write(f"Mean query execution time for sub-optimal access paths: {mean_query_execution_time_for_sub_optimal_access_path}\n")
        f.write(f"Percentiles of p_error\n")
        for percentile in percentiles_values:
            value = np.percentile(p_errors, percentile)
            print(f"Percentile ({percentile}th): {value}")
            f.write(f"{percentile}th percentile: {value}\n")
        f.write("\n")

        # "Calculate the percentiles of query execution times of sub optimal plans"
        # # sub_optimal_query_execution_time = query_execution_time_estimates[sub_optimal_access_path_id_list]
        # sub_optimal_query_planning_time = df_estimates["query_planning_time"].values[sub_optimal_access_path_id_list]
        # sub_optimal_query_total_time = df_estimates["query_total_time"].values[sub_optimal_access_path_id_list]
        # # for id in sub_optimal_access_path_id_list:
        # #     sub_optimal_query_execution_time.append(query_execution_time_estimates[id])

        # exec_time_txt = ""
        # plan_time_txt = ""
        # total_time_txt = ""
        # for percentile in percentiles_values:
        #     # value = np.percentile(sub_optimal_query_execution_time, percentile)
        #     txt = f"Percentile ({percentile}th) of query execution time for sub-optimal access paths: {value}\n"
        #     exec_time_txt += txt

        #     value = np.percentile(sub_optimal_query_planning_time, percentile)
        #     txt = f"Percentile ({percentile}th) of query planning time for sub-optimal access paths: {value}\n"
        #     plan_time_txt += txt

        #     value = np.percentile(sub_optimal_query_total_time, percentile)
        #     txt = f"Percentile ({percentile}th) of query total time for sub-optimal access paths: {value}\n"
        #     total_time_txt += txt

        # # print(exec_time_txt)
        # # print(plan_time_txt)
        # print(total_time_txt)
        # f.write(exec_time_txt)
        # f.write("\n")
        # f.write(plan_time_txt)
        # f.write("\n")
        # f.write(total_time_txt)

        # "Calculate the percentiles of query planning times of sub optimal plans"
        # sub_optimal_query_planning_time = query_execution_time_true[sub_optimal_access_path_id_list]
        # for percentile in percentiles_values:
            
        #     print(f"Percentile ({percentile}th) of query execution time for sub-optimal access paths: {value}")
        #     f.write(f"{percentile}th percentile of query execution time for sub-optimal access paths: {value}\n")
        # f.write("\n")

        # Rearrange the sub_optimal_query_execution_time in descending order and get the corresponding ids
        # sorted_indices = np.argsort(sub_optimal_query_execution_time)[::-1]
        # sorted_ids = [sub_optimal_access_path_id_list[i] for i in sorted_indices]
        # sorted_execution_times = [sub_optimal_query_execution_time[i] for i in sorted_indices]

        # Print the queries with their ids and execution times
        # print("Sub-optimal queries sorted by execution time (descending):")
        # for query_id, execution_time in zip(sorted_ids[:10], sorted_execution_times[:10]):
        #     # print(f"Query: {df_estimates['query'][query_id]}, Execution Time: {execution_time}")
        #     f.write(f"Query: {df_estimates['query'][query_id]}, Execution Time: {execution_time}\n")

    print(f"Same access paths: {(same_access_paths)}")
    # print(f"High execution time: {sum(high_execution_time)}")
    # print(f"Mean query execution time: {np.mean(query_execution_time_estimates)}")
    print("========================================================================")



def calculate_p_error_for_db(database_name, csv_file_path="scripts/plan_cost"):
    dir_path = Path(f"{csv_file_path}/{database_name}")
    if not dir_path.exists():
        return -1
    
    files = list(dir_path.glob("*true_card_cost.csv"))
    assert len(files) == 1
    true_cost_file_path = files[0]

    all_dict_list = []
    for estimates_cost_file_path in dir_path.glob("*.csv"):
        if true_cost_file_path == estimates_cost_file_path:
            continue
        print(f"Calculating p_error for {estimates_cost_file_path}")

        calculate_p_error(estimates_cost_file_path.as_posix(), true_cost_file_path.as_posix())   

if __name__ == "__main__":
    # estimates_cost_file_path = "/home/titan/phd/megadrive/End-to-End-CardEst-Benchmark/plan costs/census/census_cost_mscn.xlsx"
    # true_cost_file_path = "/home/titan/phd/megadrive/End-to-End-CardEst-Benchmark/plan costs/census/census_cost_true.xlsx"
    # calculate_p_error(estimates_cost_file_path, true_cost_file_path)


    parser = argparse.ArgumentParser(description="Execute SQL queries and export cost estimates.")
    parser.add_argument("--database_name", default="tpch_sf2_z1", type=str, help="The path to the estimates cost file.")
    parser.add_argument("--csv_file_path", default="scripts/plan_cost", type=str, help="The path to the true cost file.")
    args = parser.parse_args()
    calculate_p_error_for_db(args.database_name, args.csv_file_path)
    # estimates_cost_file_path = args.estimates_cost_file_path
    # true_cost_file_path = args.true_cost_file_path
    
