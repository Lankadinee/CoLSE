import numpy as np
from colse.cdf_base import CDFBase
from colse.emphirical_cdf import EMPMethod
from loguru import logger
from scipy.stats import rankdata

from colse.dtype_conversion import convert_to_low_precision_dtype

MAX_UNIQUE_VALUES = 25000

class OptimizedEmpiricalCDFModel(CDFBase):
    def __init__(self, **kwargs):
        self._unique_values = None
        self.cum_sum_array = None

        self._rank = None
        self._length = None
        self._query_close_to = "NA"
        self._current_index = -1
        self._rank_method = "max"
        self._emp_method = EMPMethod(kwargs.get("emp_method", "closest"))
        self._enable_low_precision = kwargs.get("enable_low_precision", False)
        self._verbose = kwargs.get("verbose", True)
        (
            logger.info(f"Empirical CDF Method: {self._emp_method} \nLow Precision: {self._enable_low_precision}")
            if self._verbose
            else None
        )
        self.max_unique_values = kwargs.get("max_unique_values", np.inf)
        logger.info(f"Max unique values: {self.max_unique_values}")

    def get_model_size(self):
        cum_sum_dict_size = self.cum_sum_array.nbytes
        unique_values_size = self._unique_values.nbytes
        # logger.info(f"Model size: {cum_sum_dict_size + unique_values_size} bytes")
        return cum_sum_dict_size + unique_values_size

    def fit(self, data):
        # Get unique values and their counts
        self._unique_values, _value_counts = np.unique(data, return_counts=True)
        initial_length = np.sum(_value_counts)
        current_unique = len(_value_counts)
        if self.max_unique_values == "auto":
            self.max_unique_values = np.clip(min(int(current_unique*0.1), MAX_UNIQUE_VALUES), current_unique, MAX_UNIQUE_VALUES)
            logger.info(f"New max unique values: {self.max_unique_values}")

        if current_unique > self.max_unique_values:
            """randomly sample indexes within current_length"""
            indexes = np.random.choice(
                np.arange(current_unique), self.max_unique_values, replace=False, p=_value_counts / np.sum(_value_counts)
            )
            self._unique_values = self._unique_values[indexes]
            _value_counts = _value_counts[indexes]
            logger.info(f"Reduced unique values from {current_unique} to {self.max_unique_values}")

        self._length = np.sum(_value_counts) + 1
        self._rank = self._empirical_cdf(self._unique_values)

        _unique_len = len(self._unique_values)
        self._unique_value_count_ = _unique_len
        (
            logger.info(
                f"Reduced values from {initial_length} to {_unique_len} | {(initial_length)/(_unique_len):.1f}X times reduction"
            )
            if self._verbose
            else None
        )
        cumilative_sum = 0

        self.cum_sum_array = np.zeros_like(self._rank, dtype=np.int32)
        for arg_index in np.argsort(self._rank):
            cumilative_sum += _value_counts[arg_index]
            self.cum_sum_array[arg_index] = cumilative_sum
        
        if self._enable_low_precision:
            self.cum_sum_array = convert_to_low_precision_dtype(self.cum_sum_array)
            self._unique_values = convert_to_low_precision_dtype(self._unique_values)

    def _get_length(self, data):
        if isinstance(data, list):
            length = len(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                length = len(data)
            else:
                length = max(data.shape[1], data.shape[0])
        return length + 1

    def _empirical_cdf(self, data):
        return rankdata(data, method=self._rank_method)

    def _get_query_side(self):
        return self._query_close_to

    def _get_closest_index(self, value):
        # Find the index of the closest value in the column self.X[col_no]
        closest_index = np.abs(self._unique_values - value).argmin()

        closest_value = self._unique_values[closest_index]
        self._query_close_to = "left" if closest_value > value else "right"
        return closest_index

    def _get_closest_positive_and_negative_index(self, value):
        data_points = self._unique_values - value
        positive_data_points = data_points.copy()
        positive_data_points = np.where(data_points >= 0, data_points, np.inf)
        negative_data_points = data_points.copy()
        negative_data_points = np.where(data_points <= 0, data_points, -np.inf)
        positive_index = np.abs(positive_data_points).argmin()
        negative_index = np.abs(negative_data_points).argmin()
        return positive_index, negative_index

    def _get_cdf(self, index):
        return self.cum_sum_array[index] / self._length

    def _get_previous_cdf(self, index):
        current_rank_value = self._rank[index]
        less_than_current_rank = self._rank[self._rank < current_rank_value]
        no_of_values_less_than_current_rank = len(less_than_current_rank)
        return no_of_values_less_than_current_rank / self._length

    def _get_current_index(self):
        return self._current_index

    def predict(self, value):
        if value == np.inf:
            return 1
        elif value == -np.inf:
            return 0
        else:
            if self._emp_method == EMPMethod.CLOSEST:
                self._current_index = self._get_closest_index(value)
                return self._get_cdf(self._current_index)
            elif self._emp_method == EMPMethod.RELATIVE:
                positive_index, negative_index = (
                    self._get_closest_positive_and_negative_index(value)
                )
                if positive_index == negative_index:
                    return self._get_cdf(positive_index)
                positive_cdf = self._get_cdf(positive_index)
                negative_cdf = self._get_cdf(negative_index)
                relative_cdf = negative_cdf + (positive_cdf - negative_cdf) * (
                    value - self._unique_values[negative_index]
                ) / (
                    self._unique_values[positive_index]
                    - self._unique_values[negative_index]
                )
                return relative_cdf
            else:
                raise ValueError("Invalid emp_method")

    def get_range_cdf(self, lb, ub):
        assert (
            self._emp_method == EMPMethod.CLOSEST
        ), "This method is only supported for emp_method='closest'"
        assert True, "This method is not properly implemented yet"
        if lb == -np.inf and ub == np.inf:
            return 1

        if lb == -np.inf:
            cdf_lb = 0
            cdf_index_lb = -1
            cdf_side_lb = "NA"
        else:
            cdf_lb = self.predict(lb)
            cdf_index_lb = self._get_current_index()
            cdf_side_lb = self._get_query_side()

        if ub == np.inf:
            cdf_ub = 1
            cdf_index_ub = -1
            cdf_side_ub = "NA"
        else:
            cdf_ub = self.predict(ub)
            cdf_index_ub = self._get_current_index()
            cdf_side_ub = self._get_query_side()

        if cdf_index_lb == cdf_index_ub:
            if cdf_side_lb == cdf_side_ub:
                return 0
            else:
                assert cdf_lb == cdf_ub
                return cdf_lb - self._get_previous_cdf(cdf_index_lb)
        else:
            if cdf_side_lb == "left":
                left_cdf = self._get_previous_cdf(cdf_index_lb)
            else:
                left_cdf = cdf_lb

            if cdf_side_ub == "left":
                right_cdf = self._get_previous_cdf(cdf_index_ub)
            else:
                right_cdf = cdf_ub

            return right_cdf - left_cdf


if __name__ == "__main__":
    model = OptimizedEmpiricalCDFModel(emp_method=EMPMethod.CLOSEST, max_unique_values=5000)
    mock_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 100000)
    """mock data using random choice"""
    np.random.seed(42)
    # mock_data = np.random.choice([(v + 1) * 1 for v in range(100)], 300)
    mock_data = np.random.random(10000)
    model.fit(mock_data)
    # print(model._rank)
    """check whether its clos to this value"""
    # assert abs(model.predict(54) - 0.495) < 1e-4

    value = 54.5
    logger.info(f"Model prediction: {model.predict(value)}")

    def my_func():
        model.predict(value)

    logger.info(f"{model.get_model_size()} bytes")
    import timeit

    iterations = 100
    logger.info(timeit.timeit(my_func, number=iterations) / iterations)

    """
    2025-02-08 17:48:05.682 | INFO     | __main__:__init__:21 - Empirical CDF Method: EMPMethod.CLOSEST
    2025-02-08 17:48:05.710 | INFO     | __main__:fit:42 - Reduced values from 300 to 97 | 3X times reduction
    2025-02-08 17:48:05.710 | INFO     | __main__:<module>:186 - Model prediction: 0.5348837209302325
    2025-02-08 17:48:05.710 | INFO     | __main__:<module>:191 - 1164 bytes
    2025-02-08 17:48:05.712 | INFO     | __main__:<module>:195 - 1.2493570102378726e-05
    ---------------------------------------------------------------------------------------
    2025-02-08 17:48:29.904 | INFO     | __main__:__init__:21 - Empirical CDF Method: EMPMethod.RELATIVE
    2025-02-08 17:48:29.930 | INFO     | __main__:fit:42 - Reduced values from 300 to 97 | 3X times reduction
    2025-02-08 17:48:29.930 | INFO     | __main__:<module>:186 - Model prediction: 0.5382059800664452
    2025-02-08 17:48:29.931 | INFO     | __main__:<module>:191 - 1164 bytes
    2025-02-08 17:48:29.933 | INFO     | __main__:<module>:195 - 2.4672929430380464e-05
    """
