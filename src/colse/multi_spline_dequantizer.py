import numpy as np
import pandas as pd



from loguru import logger
from colse.data_conversion_params import DataConversionParamValues, DataConversionParams
from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.join_dataset_info import get_all_columns, get_query_indexes, get_table_cols
from colse.df_utils import load_dataframe
from colse.spline_dequantizer import SplineDequantizer


class MultiSplineDequantizer:
    def __init__(self, dataset_type: DatasetNames):
        assert dataset_type.is_join_type(), "MultiSplineDequantizer only supports IMDB and CUSTOM_JOIN datasets"
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
        self.dc_param_values : dict[str, DataConversionParamValues] = {}

    
    def fit_transform(self):
        for table_name, dequantizer in self.dequantizers.items():
            dataset_path = get_data_path(self.dataset_type) / f"{table_name}.parquet"
            df = load_dataframe(dataset_path)
            non_continuous_columns = self.dataset_type.get_non_continuous_columns(table_name=table_name)
            dequantizer.fit(df, columns=non_continuous_columns)
            if len(non_continuous_columns) > 0:
                logger.info(f"Dequantizing {len(non_continuous_columns)} non-continuous columns")
                df = dequantizer.transform(df, table_name=table_name)
            else:
                logger.info("No non-continuous columns to dequantize")

            dc_params = DataConversionParams(self.dataset_type)
            dc_params_values = dc_params.store_data_conversion_params_df(df)
            self.dc_param_values[table_name] = dc_params_values
            

    def get_converted_cdf(self, table_name, query):
        no_of_cols = len(get_table_cols(self.dataset_type)[table_name])
        return self.dequantizers[table_name].get_converted_cdf(query, column_indexes=[i for i in range(no_of_cols)], table_name=table_name)
    
    def get_full_cdf(self, full_query: np.ndarray):
        full_query = full_query[0]
        all_tables = self.dataset_type.get_join_tables()
        full_cdf = np.zeros(full_query.shape[0])
        for table_name in all_tables:
            indexes = get_query_indexes(self.dataset_type, table_name)
            query = full_query[indexes]
            dummy_query = np.concatenate(([0, 0], query))
            cdf = self.get_converted_cdf(table_name, dummy_query)[2:]
            full_cdf[indexes] = cdf
        return full_cdf


    def get_mapped_normalized_query(self, full_query):
        full_query = full_query[0]
        all_tables = self.dataset_type.get_join_tables()
        full_mapped_query = full_query.copy()
        for table_name in all_tables:
            indexes = get_query_indexes(self.dataset_type, table_name)
            query = full_query[indexes]
            column_indexes = [i + 1 for i in range(len(indexes)//2)]
            mapped_query =  self.dequantizers[table_name].get_mapped_query(query.reshape(1, -1), column_indexes=column_indexes, table_name=table_name)
            dummy_query = [0, 0] + list(mapped_query)
            dummy_query = self.dc_param_values[table_name].get_min_max_normalized_lb_ub(dummy_query)[2:]
            full_mapped_query[indexes] = dummy_query
        return full_mapped_query