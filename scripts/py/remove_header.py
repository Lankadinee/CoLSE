

import os

import pandas as pd
from loguru import logger
from pathlib import Path

source_ext = ".csv"



def remove_header_if_exists(source_file_path: str):
    df = pd.read_csv(source_file_path, header=None)
    # check first rwo contains string values
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        # move original file to a new name
        backup_path = source_file_path.replace(source_ext, ".csv.backup")
        print(f"Moving {source_file_path} to {backup_path}")
        os.rename(source_file_path, backup_path)
        # save without header
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df.to_csv(source_file_path, index=False, header=False)
        return True
    else:
        return False

def convert_df(source_dir: str):
    for file in os.listdir(source_dir):
        if file.endswith(source_ext):
            source_file_path = os.path.join(source_dir, file)

            is_removed = remove_header_if_exists(source_file_path)
            if is_removed:
                logger.info(f"File {file} has been processed.")
            else:
                logger.info(f"File {file} does not contain string values in the first row, skipping.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remove header from CSV files in a directory.")
    parser.add_argument("--source_dir", type=str, help="Source directory containing CSV files.")
    args = parser.parse_args()
    source_dir = args.source_dir
    convert_df(source_dir=source_dir)


def test():
    convert_df(source_dir="workloads/forest/estimates_test")

if __name__ == "__main__":
    main()