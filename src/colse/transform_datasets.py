import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from loguru import logger

from colse.cat_transform import DeQuantize
from colse.df_utils import load_dataframe, save_dataframe
from colse.spline_dequantizer import SplineDequantizer


def dequantize_column(args):
    df, cols = args
    logger.info(f"Dequantizing column {cols}")
    start_time = datetime.now()
    dequantize = DeQuantize()

    values = dequantize.fit_transform(df[f"{cols}"].to_numpy())
    logger.info(f"Dequantized column {cols} in {datetime.now() - start_time}")
    return cols, values


def convert_df_to_dequantize(df, parellel, col_list_to_be_dequantized=None):
    logger.info(f"DF: {df.shape}")
    quant_dict = DeQuantize.get_dequantizable_columns(df, col_list_to_be_dequantized)
    iterable = [(df, cols) for cols in df.columns if quant_dict[cols].is_dequantizable]

    if parellel:
        with Pool() as pool:
            results = pool.map(dequantize_column, iterable)
    else:
        results = [dequantize_column(i) for i in iterable]

    result_dict = {cols: values for cols, values in results}

    new_df = pd.DataFrame()
    for col_name in df.columns:
        if not quant_dict[col_name].is_dequantizable:
            new_df[f"{col_name}"] = df[f"{col_name}"]
        else:
            new_df[f"{col_name}"] = result_dict[col_name]

    return new_df


def dataset_conversio_v2(excel_path):
    start_time = time.perf_counter()
    deq = SplineDequantizer(m=10000)

    # excel_path = Path("library/data/power/original.csv")
    df = load_dataframe(excel_path)

    print(df.describe())
    print(df.head())

    # 3) Fit on the columns you want to dequantize
    deq.fit(df)

    # 4) Transform (dequantize) those columns into continuous [0,1) arrays
    df_cont = deq.transform(df)
    print(df_cont.describe())

    logger.info(f"Time taken: {time.perf_counter() - start_time}")

    start_time = time.perf_counter()
    save_dataframe(
        df_cont, excel_path.parent / f"{excel_path.stem}_dequantized_v2.parquet"
    )
    logger.info(f"Time taken for saving: {time.perf_counter() - start_time}")


def power_dataset_conversion():
    start_time = time.time()
    excel_path = Path("library/data/power/original.csv")
    rows_to_read = None
    df = load_dataframe(excel_path)

    trimmed_df = df[:rows_to_read] if rows_to_read else df
    if rows_to_read:
        logger.info(f"Trimmed DF: {trimmed_df.shape}")
        df_path = excel_path.parent / f"{excel_path.stem}_trimmed_{rows_to_read}.csv"
        save_dataframe(trimmed_df, df_path)
        dfq_path = (
            excel_path.parent / f"{excel_path.stem}_dequantized_{rows_to_read}.csv"
        )
    else:
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized_all.csv"

    new_df = convert_df_to_dequantize(
        trimmed_df, parellel=True, col_list_to_be_dequantized=list(df.columns)
    )
    save_dataframe(new_df, dfq_path)
    logger.info(f"Time taken: {time.time() - start_time}")


def dmv_dataset_conversion():
    start_time = time.time()
    excel_path = Path("library/data/dmv/dmv.csv")
    rows_to_read = None
    df = load_dataframe(excel_path)

    post_fix = "_10KU"
    trimmed_df = df[:rows_to_read] if rows_to_read else df
    if rows_to_read:
        logger.info(f"Trimmed DF: {trimmed_df.shape}")
        df_path = (
            excel_path.parent
            / f"{excel_path.stem}_trimmed_{rows_to_read}{post_fix}.csv"
        )
        save_dataframe(trimmed_df, df_path)
        dfq_path = (
            excel_path.parent
            / f"{excel_path.stem}_dequantized_{rows_to_read}{post_fix}.csv"
        )
    else:
        dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized{post_fix}.csv"

    new_df = convert_df_to_dequantize(
        trimmed_df, parellel=True, col_list_to_be_dequantized=None
    )

    # new_df = trimmed_df.copy()
    # col, values = dequantize_column((trimmed_df, 'Reg_Valid_Date'))
    # new_df[col] = values

    save_dataframe(new_df, dfq_path)
    logger.info(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":
    # power_dataset_conversion()
    dataset_conversio_v2(Path("data/power/original.csv"))
