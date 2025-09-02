



from loguru import logger
from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.dataset_imdb import TABLE_COLS
from colse.df_utils import load_dataframe
from colse.spline_dequantizer import SplineDequantizer


class MultiSplineDequantizer:
    def __init__(self, dataset_type: DatasetNames):
        assert dataset_type == DatasetNames.IMDB_DATA, "MultiSplineDequantizer only supports IMDB dataset"
        self.dataset_type = dataset_type
        table_names = dataset_type.get_join_tables()
        self.dequantizers = {
            table_name: SplineDequantizer(
                dataset_type=dataset_type,
                cache_name=f"{table_name}_dequantizer.pkl",
                output_file_name=f"{table_name}_dequantized.parquet",
                enable_uniques_shuffling=False,
            ) for table_name in table_names
        }

    
    def fit_transform(self):
        for table_name, dequantizer in self.dequantizers.items():
            dataset_path = get_data_path(self.dataset_type) / f"{table_name}.parquet"
            df = load_dataframe(dataset_path)
            non_continuous_columns = self.dataset_type.get_non_continuous_columns(table_name=table_name)
            dequantizer.fit(df, columns=non_continuous_columns)
            if len(non_continuous_columns) > 0:
                logger.info(f"Dequantizing {len(non_continuous_columns)} non-continuous columns")
                dequantizer.transform(df)
            else:
                logger.info("No non-continuous columns to dequantize")

    def get_converted_cdf(self, table_name, query):
        no_of_cols = len(TABLE_COLS[table_name])
        return self.dequantizers[table_name].get_converted_cdf(query, column_indexes=[i for i in range(no_of_cols)])
