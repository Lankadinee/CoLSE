from colse.df_utils import save_dataframe
from colse.df_utils import load_dataframe


csv_file_name = "/datadrive500/CoLSE/data/dmv/dmv.csv"
parquet_file_name = "/datadrive500/CoLSE/data/dmv/dmv.parquet"
no_of_rows = None

def csv_2_parquet():
    df = load_dataframe(csv_file_name)
    if no_of_rows is not None:
        df = df.head(no_of_rows)
    save_dataframe(df, parquet_file_name)

def parquet_2_csv():
    df = load_dataframe(parquet_file_name)
    if no_of_rows is not None:
        df = df.head(no_of_rows)
    save_dataframe(df, csv_file_name)

if __name__ == "__main__":
    csv_2_parquet()