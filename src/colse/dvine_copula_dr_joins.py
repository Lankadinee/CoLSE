import time

import numpy as np
from loguru import logger

from colse.column_index import ColumnIndexProvider, ColumnIndexTypes
from colse.copula_types import CopulaTypes
from colse.data_path import DataPathDir, get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.join_dataset_info import get_all_columns, get_no_of_cols, get_table_cols
from colse.df_utils import load_dataframe
from colse.divine_copula_dynamic_recursive import DivineCopulaDynamicRecursive
from colse.multi_spline_dequantizer import MultiSplineDequantizer
from colse.theta_storage import ThetaStorage


class MultiDivineCopulaDynamicRecursive:
    def __init__(self, dataset_type: DatasetNames, ms_dequantizer: MultiSplineDequantizer) -> None:
        self.dataset_type = dataset_type
        self.theta_dicts = {}
        self.col_index_providers = {}
        self.models = {}
        self.ms_dequantizer: MultiSplineDequantizer = ms_dequantizer
        self.query_col_list = get_all_columns(self.dataset_type)
        self.no_of_rows = {}
        self.load_initial_data()
        self.no_of_bins = 1000
        self.bin_value_list = self.split_bins()

    def split_bins(self):
        root_table_name = self.dataset_type.get_join_tables()[0] + ".parquet"
        dataset_path = get_data_path(self.dataset_type) / root_table_name
        df = load_dataframe(dataset_path)
        all_ids = df["id"].to_numpy()
        # sort the ids
        all_ids = np.sort(all_ids)
        # split into 10 bins with distinct values
        bin_size = len(all_ids) // self.no_of_bins
        bin_value_list = [all_ids[0]]
        for i in range(1, self.no_of_bins - 1):
            bin_ids = all_ids[int(i * bin_size)]
            bin_value_list.append(bin_ids)
        bin_value_list.append(all_ids[-1])
        return bin_value_list

    def load_initial_data(self):
        for table_name in self.dataset_type.get_join_tables():
            dataset_path = get_data_path(self.dataset_type) / f"{table_name}.parquet"
            df = load_dataframe(dataset_path)
            self.theta_dicts[table_name] = ThetaStorage(
                CopulaTypes.GUMBEL, get_no_of_cols(self.dataset_type)[table_name]
            ).get_theta(
                df,
                cache_name=get_data_path(
                    DataPathDir.THETA_CACHE, self.dataset_type.value
                )
                / f"{table_name}.pkl",
            )
            self.col_index_providers[table_name] = ColumnIndexProvider(
                df, ColumnIndexTypes.NATURAL_SKIP_ORDERING
            )
            self.models[table_name] = DivineCopulaDynamicRecursive(
                theta_dict=self.theta_dicts[table_name]
            )
            self.no_of_rows[table_name] = df.shape[0]

    def get_sub_queries(self, query, joined_tables):
        table_wise_sub_queries = {}
        for joined_table in joined_tables:
            table_wise_sub_queries[joined_table] = [
                "<lower movie_id>",
                "<upper movie_id>",
            ]
            for column in get_table_cols(self.dataset_type)[joined_table][1:]:
                q_col_name = f"{joined_table}:{column}"
                q_index = self.query_col_list.index(q_col_name)
                lb_query_index = q_index * 2
                ub_query_index = q_index * 2 + 1
                lb_query = query[0][lb_query_index]
                ub_query = query[0][ub_query_index]
                table_wise_sub_queries[joined_table].extend([lb_query, ub_query])

            table_wise_sub_queries[joined_table] = np.array(
                table_wise_sub_queries[joined_table]
            )
        return table_wise_sub_queries

    def modify_sub_query_for_bin(self, sub_query, l_bin, u_bin):
        copy_sub_query = sub_query.copy()
        copy_sub_query[0] = l_bin
        copy_sub_query[1] = u_bin
        return copy_sub_query

    def predict(self, query, joined_tables):
        logger.info(f"Query: {query}")
        start_time = time.time()
        joined_table_len = len(joined_tables)
        result = np.zeros((joined_table_len, self.no_of_bins))
        total_rows = 1
        for joined_table_index, (joined_table, s_query) in enumerate(
            self.get_sub_queries(query, joined_tables).items()
        ):
            for bin_index, (l_bin, u_bin) in enumerate(
                zip(self.bin_value_list, self.bin_value_list[1:])
            ):
                mod_sub_query = self.modify_sub_query_for_bin(
                    s_query, l_bin, u_bin
                )
                mod_sub_query_cdf = self.ms_dequantizer.get_converted_cdf(
                    joined_table, mod_sub_query
                )
                col_indices, cdf_list = self.col_index_providers[
                    joined_table
                ].get_column_index(mod_sub_query_cdf)

                y_bar = self.models[joined_table].predict(
                    cdf_list, column_list=col_indices
                )
                # print(
                #     f"Prediction for {joined_table} with {l_bin} and {u_bin} is {y_bar}"
                # )
                result[joined_table_index][bin_index] = y_bar

            base_s_query = s_query.copy()
            base_s_query[0] = -np.inf
            base_s_query[1] = np.inf
            base_sub_query_cdf = self.ms_dequantizer.get_converted_cdf(
                joined_table, base_s_query
            )
            col_indices, cdf_list = self.col_index_providers[
                joined_table
            ].get_column_index(base_sub_query_cdf)
            y_bar_div = self.models[joined_table].predict(
                cdf_list, column_list=col_indices
            )
            result[joined_table_index] /= y_bar_div
            total_rows *= self.no_of_rows[joined_table] * y_bar_div

        copula_pred_time = time.time() - start_time
        selectivity = np.sum(np.prod(result, axis=0, keepdims=False), axis=0)
        card_est = total_rows * selectivity
        print(f"Time Taken: {copula_pred_time} Selectivity: {selectivity} Total Rows: {total_rows} Card Est: {card_est}")
        return int(round(card_est, 0))

        # original_cdf_list = s_dequantize.get_converted_cdf(query, COLUMN_INDEXES)
        # col_indices, cdf_list = col_index_provider.get_column_index(original_cdf_list)
        # no_of_cols_for_this_query = len(col_indices)
        # loop.set_description(f"#cols: {no_of_cols_for_this_query:2d}")
        # y_bar = model.predict(cdf_list, column_list=col_indices)
