import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from colse.cat_transform import DeQuantize
from colse.df_utils import load_dataframe, save_dataframe
from loguru import logger


def dequantize_column(args):
    np_df_col, cols = args
    logger.info(f"Dequantizing column {cols}")
    start_time = datetime.now()
    dequantize = DeQuantize()
    
    values = dequantize.fit_transform(np_df_col, col_name=f"{cols}")
    logger.info(f"Dequantized column {cols} in {datetime.now() - start_time}")
    return cols, values


def convert_df_to_dequantize(df, parellel, col_list_to_be_dequantized=None, col_list_to_exclude=None):
    logger.info(f"DF: {df.shape}")
    quant_dict = DeQuantize.get_dequantizable_columns(df, col_list_to_be_dequantized, col_list_to_exclude=col_list_to_exclude)
    iterable = [(df[cols].to_numpy(), cols) for cols in df.columns if quant_dict[cols].is_dequantizable]

    if parellel:
        logger.info("Parellel Dequantization")
        with Pool() as pool:
            results = pool.map(dequantize_column, iterable)
    else:
        results = [dequantize_column(i) for i in iterable]

    logger.info(f"Dequantization completed - {len(results)} columns dequantized")
    st = datetime.now()
    result_dict = {cols: values for cols, values in results}

    new_df = pd.DataFrame()
    for col_name in df.columns:
        # logger.info(f"Processing column {col_name}")
        if not quant_dict[col_name].is_dequantizable:
            new_df[f"{col_name}"] = df[f"{col_name}"]
        else:
            new_df[f"{col_name}"] = result_dict[col_name]
    # logger.info(f"New DF Columns: {new_df.columns}")
    logger.info(f"Dequantization completed - {new_df.shape} df create time {datetime.now() - st}")
    return new_df   


def power_dataset_conversion(excel_file_path: str, deqantize_columns: list, exclude_columns: list = None):
    start_time = time.time()
    excel_path = Path(excel_file_path)
    rows_to_read = None
    df = load_dataframe(excel_path)

    trimmed_df = df[:rows_to_read] if rows_to_read else df
    if rows_to_read:
        logger.info(f"Trimmed DF: {trimmed_df.shape}")
        df_path = excel_path.parent / f"{excel_path.stem}_trimmed_{rows_to_read}.csv"
        save_dataframe(trimmed_df, df_path)
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized_{rows_to_read}.csv"
    else:
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized.csv"

    new_df = convert_df_to_dequantize(trimmed_df, parellel=False, col_list_to_be_dequantized=deqantize_columns, col_list_to_exclude=exclude_columns)
    logger.info(f"Time taken: {time.time() - start_time}")
    save_dataframe(new_df, dfq_path)


def dmv_dataset_conversion():
    start_time = time.time()
    excel_path = Path("library/data/dmv/dmv.csv")
    rows_to_read = None
    df = load_dataframe(excel_path)

    post_fix = "_10KU"
    trimmed_df = df[:rows_to_read] if rows_to_read else df
    if rows_to_read:
        logger.info(f"Trimmed DF: {trimmed_df.shape}")
        df_path = excel_path.parent / f"{excel_path.stem}_trimmed_{rows_to_read}{post_fix}.csv"
        save_dataframe(trimmed_df, df_path)
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized_{rows_to_read}{post_fix}.csv"
    else:
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized{post_fix}.csv"

    new_df = convert_df_to_dequantize(trimmed_df, parellel=True, col_list_to_be_dequantized=None)

    # new_df = trimmed_df.copy()
    # col, values = dequantize_column((trimmed_df, 'Reg_Valid_Date'))
    # new_df[col] = values

    save_dataframe(new_df, dfq_path)
    logger.info(f"Time taken: {time.time() - start_time}")

def main():
    column_list_to_dequantize = ["l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode"]
    column_list_to_exclude = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice",
                                "l_discount", "l_tax", "l_commitdate", "l_receiptdate", "l_shipdate"]
    for f_path in [#"library/data/tpch_sf2_z1_lineitem/original.csv", "library/data/tpch_sf2_z2_lineitem/original.csv",
                   #"library/data/tpch_sf2_z3_lineitem/original.csv", "library/data/tpch_sf2_z4_lineitem/original.csv", "library/data/tpch_lineitem_20/original.parquet"
                   "library/data/tpch_lineitem_20/original.parquet"]:
        start_time = time.time()
        power_dataset_conversion(excel_file_path=f_path, deqantize_columns=column_list_to_dequantize, exclude_columns=column_list_to_exclude)
        logger.info(f"Time taken for the dequantization: {time.time() - start_time}")
        logger.info(f"---------------------------------------")

if __name__ == "__main__":
    # power_dataset_conversion(excel_file_path="library/data/census/census.csv", deqantize_columns=None)
    # dmv_dataset_conversion()
    main()
        
