"""
Convert IMDB CSVs to Parquet files.

imdb DATASET LOCATION:
"""

from pathlib import Path

import pandas as pd

from colse.df_utils import save_dataframe

source_path = Path("./data")
destination_path = Path(".")

destination_path.mkdir(parents=True, exist_ok=True)

FILE_NAMES = [
    "cast_info.csv",
    "movie_companies.csv",
    "movie_info.csv",
    "movie_info_idx.csv",
    "movie_keyword.csv",
    "title.csv",
]

COL_NAMES_MAP = {
    "cast_info": ["movie_id", "role_id"],
    "movie_companies": ["movie_id", "company_id", "company_type_id"],
    "movie_info": ["movie_id", "info_type_id"],
    "movie_keyword": ["movie_id", "keyword_id"],
    "title": ["id", "kind_id", "production_year"],
    "movie_info_idx": ["movie_id", "info_type_id"],
}


def convert_to_parquet_from_csv(file_name: str, to: str = "parquet"):
    csv_path = source_path / file_name
    dest_path = destination_path / file_name.replace(".csv", f".{to}")

    print(f"Loading {csv_path}")
    df = pd.read_csv(
        csv_path, on_bad_lines="skip"
    )
    print(df.head())
    print(f"Loaded df from csv : {df.shape} rows and columns : {df.columns}")
    df = df[COL_NAMES_MAP[file_name.replace(".csv", "")]]
    print(f"Loaded {len(df)} rows")
    save_dataframe(df, dest_path)
    print(f"Saved df to {dest_path}")


def main():
    dest_to = "csv"
    for file_name in FILE_NAMES:
        convert_to_parquet_from_csv(file_name, to=dest_to)


if __name__ == "__main__":
    main()
