import subprocess

import pandas as pd


def get_file_list(container_name, directory):
    # Run docker exec command
    result = subprocess.run(
        ["docker", "exec", container_name, "ls", "-1", directory],
        capture_output=True,
        text=True
    )

    # Print or process the output
    files = []
    if result.returncode == 0:
        files = result.stdout.strip().split("\n")
        # print("Files inside the container:", files)
    else:
        print("Error:", result.stderr)
        raise Exception("Error while listing files inside the container")

    return files

def get_all_unprocessed_txt_files(container_name, directory):
    files = get_file_list(container_name, directory)
    all_txt_files = [file.split('.')[0] for file in files if file.endswith(".csv")]
    all_excel_files = [file.split('.')[0] for file in files if file.endswith(".xlsx")]
    unprocessed_files = list(set(all_txt_files) - set(all_excel_files))
    return [f"{fname}.csv" for fname in unprocessed_files]

def get_all_input_files(container_name, directory):
    files = get_file_list(container_name, directory)
    all_csv_files = [file.split('.')[0] for file in files if file.endswith(".csv")]
    return [f"{fname}.csv" for fname in all_csv_files]

def analyze_results(estimate_file_name, true_file_name):
    df_estimates = pd.read_csv(estimate_file_name)
    df_true = pd.read_csv(true_file_name)

    # """Merge the access_path column in df_true to df_estimates"""
    # df_estimates = df_estimates.merge(df_true['access_path'], on='true_access_path', how='left')

    """Create a new column 'same_access_paths' in df_estimates if the access_path in df_estimates is same as the access_path in df_true"""
    df_estimates['same_access_paths'] = df_estimates['access_path'] == df_true['access_path']

    result = df_estimates.groupby('no_queried_columns')['same_access_paths'].value_counts().unstack().fillna(0)
    # print(result)
    # rename input_card column to input_card_est
    df_estimates.rename(columns={'input_card_est': 'predicted_card_est'}, inplace=True)
    df_estimates = pd.concat([df_estimates, df_true[['input_card_est']]], axis=1)

    # for est_card, card in zip(df_estimates['input_card_est'].values, df_true['input_card_est'].values):
    def q_error(est_card, card):
        q_errors = []
        if est_card == 0 and card == 0:
            q_errors.append(1.0)
        if est_card == 0:
            q_errors.append(card)
        if card == 0:
            q_errors.append(est_card)
        if est_card > card:
            q_errors.append(est_card / card)
        else:
            q_errors.append(card / est_card)
        return q_errors[0]
    
    df_estimates["q_error"] = df_estimates.apply(lambda row: q_error(row['predicted_card_est'], row['input_card_est']), axis=1)

    # percentiles = [0.5, 0.9, 0.95, 0.99, 1.0]
    # for p in percentiles:
    #     print(f"Percentile {p}: {pd.Series(q_errors).quantile(p)}")

    # group by no_queried_columns and calculate percentiles
    percentiles = [0.5, 0.9, 0.95, 0.99, 1.0]
    qerror_df = pd.DataFrame()
    col_names = []
    for p in percentiles:
        # print(f"Percentile {p}: {df_estimates.groupby('no_queried_columns')['q_error'].quantile(p)}")
        # add these groups to one dataframe
        qerror_df = pd.concat([qerror_df, df_estimates.groupby('no_queried_columns')['q_error'].quantile(p)], axis=1)
        col_names.append(f"q_error_{p}")
    qerror_df.columns = col_names
    # qerror_df.reset_index(inplace=True)
    # qerror_df.rename(columns={'no_queried_columns': 'no_queried_columns'}, inplace=True)

    print(qerror_df)



if __name__ == "__main__":
    # unproc_files = get_all_unprocessed_txt_files("ce-benchmark-ceb1-forest", "/var/lib/pgsql/13.1/data/")
    # print("Unprocessed files:", unproc_files)

    analyze_results(estimate_file_name="scripts/plan_cost/correlated_04/mhist_30000_correlated_04_estimates_cost.csv",
                    true_file_name="scripts/plan_cost/correlated_04/correlated_04_true_card_cost.csv")