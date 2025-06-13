


import os

from colse.df_utils import load_dataframe, save_dataframe


def convert_df(source_dir: str, source_ext: str, destination_ext: str):
    for file in os.listdir(source_dir):
        if file.endswith(source_ext):
            df = load_dataframe(os.path.join(source_dir, file))
            save_dataframe(df, os.path.join(source_dir, file.replace(source_ext, destination_ext)))


if __name__ == "__main__":
    convert_df(source_dir="/home/titan/phd/CoLSE/data/census/data_updates", source_ext=".csv", destination_ext=".parquet")
