from colse.df_utils import load_dataframe
from colse.spline_dequantizer import SplineDequantizer
from colse.datasets.dataset_dmv import DatasetNames

dataset_type = DatasetNames.DMV_DATA

def main():
    # # Dequantize dataset
    # s_dequantize = SplineDequantizer()
    # dataset_path = dataset_type.get_file_path()
    # df = load_dataframe(dataset_path)
    # col_names = list(df.columns)
    # s_dequantize.fit(df, columns=dataset_type.get_non_continuous_columns())

    # pickle save dequantize object and get the file size
    import pickle
    import os
    DATASET_NAME = "power"
    pickle_file_path = f"data/cdf_cache/{DATASET_NAME}/spline_dequantizer_cache.pkl"
    file_size = os.path.getsize(pickle_file_path)
    print(f"Dequantize object size: {file_size / 1024 / 1024} mb")

if __name__ == "__main__":
    main()
