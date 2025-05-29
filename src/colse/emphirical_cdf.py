from enum import Enum

import numpy as np
from colse.cdf_base import CDFBase
from loguru import logger
from scipy.stats import rankdata


class EMPMethod(str, Enum):
    CLOSEST = "closest"
    RELATIVE = "relative"


class EmpiricalCDFModel(CDFBase):
    def __init__(self, **kwargs):
        self._data = None
        self._rank = None
        self._length = None
        self._query_close_to = "NA"
        self._current_index = -1
        self._rank_method = "max"
        self._emp_method = EMPMethod(kwargs.get("emp_method", "closest"))
        logger.info(f"Empirical CDF Method: {self._emp_method}")

    def get_model_size(self):
        data_size = self._data.nbytes
        rank_size = self._rank.nbytes
        # logger.info(f"Model size: {data_size + rank_size} bytes")
        return data_size + rank_size

    def fit(self, data):
        self._data = data
        self._rank, self._length = self._empirical_cdf(data)

    def _empirical_cdf(self, data):
        ranks = rankdata(data, method=self._rank_method)
        if isinstance(data, list):
            length = len(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                length = len(data)
            else:
                length = max(data.shape[1], data.shape[0])
        return ranks, (length + 1)

    def _get_query_side(self):
        return self._query_close_to

    def _get_closest_index(self, value):
        # Find the index of the closest value in the column self.X[col_no]
        closest_index = np.abs(self._data - value).argmin()

        closest_value = self._data[closest_index]
        self._query_close_to = "left" if closest_value > value else "right"
        return closest_index

    def _get_closest_positive_and_negative_index(self, value):
        data_points = self._data - value
        positive_data_points = data_points.copy()
        positive_data_points = np.where(data_points >= 0, data_points, np.inf)
        negative_data_points = data_points.copy()
        negative_data_points = np.where(data_points <= 0, data_points, -np.inf)
        positive_index = np.abs(positive_data_points).argmin()
        negative_index = np.abs(negative_data_points).argmin()
        return positive_index, negative_index

    def _get_cdf(self, index):
        return self._rank[index] / self._length

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
                positive_index, negative_index = self._get_closest_positive_and_negative_index(value)
                if positive_index == negative_index:
                    return self._get_cdf(positive_index)
                positive_cdf = self._get_cdf(positive_index)
                negative_cdf = self._get_cdf(negative_index)
                relative_cdf = negative_cdf + (positive_cdf - negative_cdf) * (value - self._data[negative_index]) / (
                    self._data[positive_index] - self._data[negative_index]
                )
                return relative_cdf
            else:
                raise ValueError("Invalid emp_method")

    def get_range_cdf(self, lb, ub):
        assert self._emp_method == EMPMethod.CLOSEST, "This method is only supported for emp_method='closest'"

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
    model = EmpiricalCDFModel(emp_method=EMPMethod.CLOSEST)
    # mock_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 100000)
    np.random.seed(42)
    mock_data = np.random.choice(range(1000), 1000000)
    model.fit(mock_data)
    # print(model._rank)
    """check whether its clos to this value"""
    # assert abs(model.predict(54) - 0.495) < 1e-4

    value = 54.5
    print(f"Model prediction: {model.predict(value)}")

    def my_func():
        model.predict(value)

    import timeit

    iterations = 100
    print(timeit.timeit(my_func, number=iterations) / iterations)
