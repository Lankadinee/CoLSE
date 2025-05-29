from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(42)


class DType(Enum):
    UP = 1
    DOWN = 2
    RANDOM = 3
    CONFUSED = 4


class SampleDataGenerator:
    def __init__(self, min=1, max=100, sigma=5, data_type="int"):
        self._min = min
        self._max = max
        self._sigma = sigma
        self.df = None
        self.data_type = data_type

    def get_data(self, n, d_type):
        match d_type:
            case DType.UP:
                return np.linspace(self._min, self._max, n)
            case DType.DOWN:
                return np.linspace(self._max, self._min, n)
            case DType.RANDOM:
                return (
                    np.random.randint(self._min, self._max, n)
                    if self.data_type == "int"
                    else np.random.uniform(self._min, self._max, n)
                )
            case DType.CONFUSED:
                type_2_1 = np.linspace(self._min, int(self._max / 2), int(n / 2))
                type_2_2 = np.linspace(int(self._max / 2), self._min, int(n / 2))
                type_confused = np.concatenate((type_2_1, type_2_2))
                return type_confused
            case _:
                raise ValueError("Invalid data type")

    def generate(self, d_type_list, n):
        assert self.df is None, "Dataframe already exists. Please create a new instance of SampleDataGenerator"

        dataset = {
            f"column{idx+1}": (
            self.get_data(n, d_type) + np.random.normal(0, self._sigma, n)
            ).clip(self._min, self._max).astype(int if self.data_type == "int" else float)
            for idx, d_type in enumerate(d_type_list)
        }
        self.df = pd.DataFrame(dataset)
        return self.df

    # def q_generate(self, column_name, no_of_queries):
    #     assert self.df is not None, "Dataframe does not exist. Please create a dataframe using generate method"
    #     assert column_name in self.df.columns, f"Column {column_name} does not exist in the dataframe"

    #     sample_data = self.generate_queries(no_of_queries)
    #     true_ce = self.get_true_cardinality(sample_data, column_name)
    #     return sample_data, true_ce

    def q_generate(self, column_names, no_of_queries, range=True, remove_zeros=False):
        assert self.df is not None, "Dataframe does not exist. Please create a dataframe using generate method"
        for column_name in column_names:
            assert column_name in self.df.columns, f"Column {column_name} does not exist in the dataframe"

        sample_datas = np.array([self.generate_queries(no_of_queries, range=range) for _ in column_names])
        sample_datas = np.moveaxis(sample_datas, 0, 1)
        sample_datas = sample_datas.reshape(no_of_queries, -1)
        true_ce = np.array(
            [
                self.get_true_cardinality(sample, column_names, range=range)
                for sample in tqdm(sample_datas, total=no_of_queries)
            ]
        ) / len(self.df)

        if remove_zeros:
            sample_datas = sample_datas[true_ce != 0]
            true_ce = true_ce[true_ce != 0]
            
        return sample_datas, true_ce

    def get_true_cardinality(self, data_samples, column_names, range=True):
        de_copy = self.df.copy()
        if range:
            col_wise_data = data_samples.reshape(-1, 2)

            for idx, (lb, ub) in enumerate(col_wise_data):
                de_copy = de_copy[(de_copy[column_names[idx]] <= ub) & (de_copy[column_names[idx]] >= lb)]
        else:
            for idx, bound in enumerate(data_samples):
                de_copy = de_copy[de_copy[column_names[idx]] <= bound]

        return len(de_copy)

    def generate_queries(self, no_of_queries, range=True):
        sample_data = np.random.uniform(self._min, self._max, size=no_of_queries * 2 if range else no_of_queries)
        if not range:
            return sample_data.clip(self._min, self._max).astype(int if self.data_type == "int" else float)
        sample_data = sample_data.clip(self._min, self._max).reshape(-1, 2)
        data_samples = np.sort(sample_data, axis=1).astype(int if self.data_type == "int" else float)
        return data_samples

    # def get_true_cardinality(self, data_samples, column_name):
    #     total_values = len(self.df)
    #     true_ce = np.array([np.sum((self.df[column_name] <= ub) & (self.df[column_name] >= lb))/total_values for lb, ub in data_samples])
    #     return true_ce


if __name__ == "__main__":
    sample_data = SampleDataGenerator()
    df = sample_data.generate([DType.UP, DType.UP, DType.DOWN, DType.DOWN], 1000)
    print(df.head(3))
    X, y = sample_data.q_generate(["column1", "column2"], 10, range=False)
    print("X:", X)
    print("y:", y)
