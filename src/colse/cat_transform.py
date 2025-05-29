from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count

from loguru import logger
import numpy as np
import pandas as pd
from colse.spline import monotone_cubic_spline
from scipy.stats import rankdata
from tqdm import tqdm


class DeqDataTypes:
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"


@dataclass
class DequantizeData:
    is_dequantizable: bool
    data_type: DeqDataTypes


class DeQuantize:
    MAX_UNIQUE_VALUES = 2500
    PROPOTION = 0.2

    def __init__(self):
        np.random.seed(42)
        self.discrete = None
        self.spline = None
        self.inverse_spline = None
        self.cum_sum = None
        self.arg_sort = None
        self.x_ranges = None
        self.x = None
        self.y = None

        self.mapping = None
        self.unique_values = None
        self.min_value = None
        self.max_value = None

        self.dequantize_array = None
        self.data_type = None
        self.total_rows = None

        self.column_name = None

    @staticmethod
    def get_dequantizable_columns(df, col_list_to_be_dequantized=None, col_list_to_exclude=None):
        col_dict = {}
        for col in df.columns:
            deq_obj = DequantizeData(is_dequantizable=False, data_type=None)
            if col_list_to_exclude and col in col_list_to_exclude:
                col_dict[col] = deq_obj
                continue
            if col_list_to_be_dequantized and col in col_list_to_be_dequantized:
                deq_obj.is_dequantizable = True
                deq_obj.data_type = DeqDataTypes.DISCRETE
            else:
                # Check if all values in a column are integers and convert to int dtype
                if df[col].dtype == "float" and df[col].apply(float.is_integer).all():
                    df[col] = df[col].astype(int)

                if df[col].dtype in ["object", "bool"]:
                    deq_obj.is_dequantizable = True
                    deq_obj.data_type = DeqDataTypes.CATEGORICAL
                elif df[col].dtype in ["int64"]:
                    unique_values = df[col].unique()
                    unique_values.sort()
                    differences = np.diff(unique_values)
                    if np.percentile(differences, 95) == 1:
                        deq_obj.is_dequantizable = True
                        deq_obj.data_type = DeqDataTypes.DISCRETE

            col_dict[col] = deq_obj
        return col_dict

    def get_spline(self):
        assert self.spline is not None, "Please fit the model first"
        return self.spline

    def fit(self, X, y=None):
        start_time = datetime.now()
        """df column - Series"""
        self.total_rows = X.shape[0]
        x_unique = np.unique(X)

        if X.dtype == "object":
            """Categorical values"""
            self.discrete, unique_values = pd.factorize(X, sort=True)
            mapping = {value: code for code, value in enumerate(unique_values)}
            self.data_type = DeqDataTypes.CATEGORICAL
            self.unique_values = unique_values
        else:
            """Discrete values"""
            self.discrete = X
            mapping = {val: val for idx, val in enumerate(x_unique)}
            self.data_type = DeqDataTypes.DISCRETE
            self.unique_values = x_unique
            self.min_value = np.min(x_unique)
            self.max_value = np.max(x_unique)
        """mapping - unique values and their related values"""

        """Append another max value to the array, to avoid maping the multiple last values to the same value"""
        max = np.max(self.discrete)
        diff_max_two_numbers = max - np.max(self.discrete[self.discrete < max])
        self.discrete = np.append(self.discrete, max + diff_max_two_numbers)

        self.cum_sum = rankdata(self.discrete, method="min")
        arg_sort = np.argsort(self.cum_sum)

        self.x = np.unique(self.discrete[arg_sort])
        self.y = np.unique(self.cum_sum[arg_sort])

        assert self.x.shape[0] > 1, "The unique values are less than 2"
        assert self.y.shape[0] > 1, "The unique values are less than 2"

        for key, val in mapping.items():
            x_index = np.where(self.x == val)[0]
            value_range = (val, self.x[x_index + 1][0])
            mapping[key] = value_range

        self.arg_sort = arg_sort
        self.mapping = mapping

        time_taken = datetime.now() - start_time
        logger.info(f"Fit completed - {self.column_name} in {time_taken}")
        return self

    def transform(self, col_name=None):
        self.spline, self.inverse_spline, _ = monotone_cubic_spline(self.x, self.y)

        start_values = self.cum_sum[self.arg_sort[:-1]]
        end_values = self.cum_sum[self.arg_sort[1:]]
        diffs = end_values - start_values

        recreated_array = np.full(len(self.arg_sort[:-1]), -1, dtype=np.float64)
        loop = tqdm(zip(start_values[diffs > 0], end_values[diffs > 0]), total=len(start_values[diffs > 0]))
        processed_row_count = 0
        for start_value, end_value in loop:
            diff = end_value - start_value
            loop.set_description(f"processing {col_name} diff[{diff}]: {processed_row_count//1000}K/{self.total_rows//1000}K")

            diff_rv_count = max(int(diff * self.PROPOTION), min(diff, 10))
            random_value_count = diff_rv_count if diff_rv_count < self.MAX_UNIQUE_VALUES else self.MAX_UNIQUE_VALUES
            random_values = np.random.uniform(start_value, end_value, random_value_count)
            st = datetime.now()
            x_values = [self.inverse_spline(v) for v in random_values]
            tt = datetime.now() - st
            # logger.info(f"Inverse spline - {self.column_name} value count {random_value_count} time > {tt}")
            if diff > random_value_count:
                x_values = list(np.random.choice(x_values, diff, replace=True))
            """add x values to the recreated array to recreate the original array"""
            indices = np.where(self.cum_sum == start_value)[0]
            recreated_array[indices] = x_values
            processed_row_count += len(x_values)


        self.dequantize_array = recreated_array
        logger.info(f"Dequantization completed - {self.column_name} recreated array shape: {self.dequantize_array.shape}")
        return self.dequantize_array

    def fit_transform(self, X, y=None, col_name=None):
        self.column_name = col_name
        return self.fit(X).transform(col_name)

    def get_mapping(self, value):
        if value is np.inf or value is -np.inf:
            return value


        if self.data_type == DeqDataTypes.CATEGORICAL:
            return self.mapping[value][1]
        else:
            value = np.float64(value)
            if value <= self.min_value:
                key = self.min_value
            elif value >= self.max_value:
                key = self.max_value
            else:
                data_points = self.unique_values - value
                positive_data_points = data_points.copy()
                " We think that all the queries are 'less than or equal' or 'greater than or equal' to value"
                positive_data_points = np.where(data_points <= 0, data_points, -np.inf)
                positive_index = positive_data_points.argmax()
                key = self.unique_values[positive_index]
            return self.mapping[key][1]


# def main2():
#     excel_path = Path("dinee/megadrive/query-optimization-methods/library/cest/transform/data.xlsx")
#     rows_to_read = None
#     df = load_dataframe(excel_path)

#     trimmed_df = df[:rows_to_read] if rows_to_read else df
#     if rows_to_read:
#         logger.info(f"Trimmed DF: {trimmed_df.shape}")
#         df_path = excel_path.parent / f"{excel_path.stem}_trimmed_{rows_to_read}.csv"
#         save_dataframe(trimmed_df, df_path)
#         dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized_{rows_to_read}.csv"
#     else:
#         dfq_path = excel_path.parent / f"{excel_path.stem}_dequantized.xlsx"

#     new_df = convert_df_to_dequantize(trimmed_df, parellel=False)
#     save_dataframe(new_df, dfq_path)


def test_dquantize():
    """write a dataframe with different data types"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [1.1, 2.1, 3.1, 4.1, 5.1],
            "D": [True, False, True, False, True],
        }
    )

    deq_dict = DeQuantize.get_dequantizable_columns(df)
    print(deq_dict)


def test_cont_dequantize():
    df = pd.DataFrame(
        {
            "C": [1.12, 4, 21, 2.31, 6.22, 3.11, 8.11, 4.13, 7.22, 5.41],
        }
    )

    dq = DeQuantize()
    dq.fit(df["C"].to_numpy())

    print(dq.mapping)
    # print(dq.get_mapping(2))
    # print(dq.get_mapping(1))
    # print(dq.get_mapping(35))
    print(dq.get_mapping(3.11))


if __name__ == "__main__":
    test_cont_dequantize()
