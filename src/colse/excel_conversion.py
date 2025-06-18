

from pathlib import Path

from loguru import logger
import pandas as pd


def merge_excel_files(output_file_path):
    no_of_rows_file_2 = 9000
    no_of_rows_file_1 = 1000

    file_paht_1 = Path("/datadrive500/CoLSEL/data/excels/dvine_v1_dmv_train_sample.xlsx")
    file_paht_3 = Path("/datadrive500/CoLSEL/data/excels/dvine_v1_dmv_train_sample_retrained_ind_0.2.xlsx")

    df_1 = pd.read_excel(file_paht_3)
    logger.info(f"df_1: {df_1.shape}")
    df_2 = pd.read_excel(file_paht_1)
    logger.info(f"df_2: {df_2.shape}")

    df_1 = df_1.iloc[:no_of_rows_file_1]
    df_2 = df_2.iloc[:no_of_rows_file_2]
    logger.info(f"Combining the dataframes with the first {no_of_rows_file_2} rows of df_1 and the first {no_of_rows_file_1} rows of df_2")

    combined_df = pd.concat([df_2, df_1])
    # No need to reshape a DataFrame; ensure the index is reset after concat
    combined_df = combined_df.reset_index(drop=True)
    logger.info(f"combined_df: {combined_df.shape}")

    combined_df.to_excel(output_file_path, index=False)

def validate_excel(file_path):
    df = pd.read_excel(file_path)
    for index, value in df["mapped_query"].items():
        v_len = len(value.split(","))
        assert v_len == 22, f"Query length mismatch: {v_len} != 22 at index {index}, value: {value}"
    logger.info("Excel file validated successfully")


if __name__ == "__main__":
    output_file_path = Path("/datadrive500/CoLSEL/data/excels/dvine_v1_dmv_train_sample_combined_9_1.xlsx")
    merge_excel_files(output_file_path)
    validate_excel(output_file_path)