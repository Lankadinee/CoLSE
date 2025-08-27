



import numpy as np
from colse.dataset_names import DatasetNames
from colse.df_utils import load_dataframe


class FanoutScaling:
    def __init__(self, dataset_type: DatasetNames):
        self._dataset_type = dataset_type
        self._in_data = None
        self._fanout_data = None
        self._in_tables = None
        self._fanout_tables = None
        self._total_rows = None
        self._root_table = None
        self._queried_table_columns = None
        self._fanout_unique_values = None

        if self._dataset_type.is_join_type():
            self.load_data()

    def load_data(self):
        original_file_name = self._dataset_type.get_file_path(filename="samples.csv")
        df = load_dataframe(self._dataset_type.get_file_path(original_file_name))
        columns = df.columns.tolist()

        ori_in_tables = [col for col in columns if col.startswith("__in_")]
        ori_fanout_tables = [col for col in columns if col.startswith("__fanout_")]
        queried_table_columns = [col for col in columns if col not in ori_in_tables and col not in ori_fanout_tables]
        self._in_data = df[ori_in_tables].to_numpy()
        self._fanout_data = df[ori_fanout_tables].to_numpy()
        self._in_tables = [n.replace("__in_", "") for n in ori_in_tables]
        self._fanout_tables = [n.replace("__fanout_", "") for n in ori_fanout_tables]
        self._total_rows = df.shape[0]
        self._root_table = "title"
        self._queried_table_columns = queried_table_columns
        self._fanout_unique_values = df[ori_fanout_tables].nunique()

    def get_fanout_data(self):
        return self._fanout_data

    def get_in_data(self):
        return self._in_data
    
    def get_queries(self, query: np.ndarray ) -> list[np.ndarray]:
        grouped_query = query.reshape(-1, 2)
        # Vectorized condition without np.vectorize
        queried_column_indexes = (grouped_query[:, 0] != -np.inf) & (grouped_query[:, 1] != np.inf)
        queried_table_names = [self._queried_table_columns[i].split(":")[0] for i in queried_column_indexes]
        
        in_query = np.zeros((len(self._in_tables), 2))
        for qtn in queried_table_names: 
            if qtn in self._in_tables:
                index = self._in_tables.index(qtn)
                in_query[index, 0] = 1
                in_query[index, 1] = 1

        fanout_query_indexes = []
        for ft in self._fanout_tables: 
            if ft not in queried_table_names:
                index = self._fanout_tables.index(ft)
                fanout_query_indexes.append(index)

        fanout_query = np.zeros((len(self._fanout_tables), 2))
        # TODO: Complete the fanout query
        return self._queried_table_columns

    def get_value(self, col_indices: list[int]):
        """
        Get all the values in the in_table list and multiply everything together and check whether it is equal to one
        For the rows that above is eaual to one get all the fanout values for non table in the table_list
        Then multiply fanout values in each row and get mean of all fanouts
        """
        if not self._dataset_type.is_join_type():
            return 1
        
        # print(col_indices)
        table_list = set([self._queried_table_columns[i-1].split(":")[0] for i in col_indices])

        in_data = self._in_data
        fanout_data = self._fanout_data
        in_tables = self._in_tables
        fanout_tables = self._fanout_tables

        in_data_multiply = np.ones(in_data.shape[0])
        for table in table_list:
            in_data_multiply *= in_data[:, in_tables.index(table)]

        row_indexes = np.where(in_data_multiply == 1)[0]

        non_table_list = [table for table in in_tables if table not in table_list and table != self._root_table]

        fanout_data_multiply = np.zeros((row_indexes.shape[0], len(non_table_list)))
        for index, non_table in enumerate(non_table_list):
            fanout_data_multiply[:, index] = fanout_data[row_indexes, fanout_tables.index(non_table)]
        fanout_data_multiply = 1/np.prod(fanout_data_multiply, axis=1)
        # fanout_data_multiply = np.mean(fanout_data_multiply, axis=0)
        fanout_data_multiply = np.sum(fanout_data_multiply, axis=0) / self._total_rows

        return fanout_data_multiply


if __name__ == "__main__":
    dataset_type = DatasetNames.IMDB_DATA
    fanout_scaling = FanoutScaling(dataset_type)
    print(fanout_scaling.get_value([1,3,4]))