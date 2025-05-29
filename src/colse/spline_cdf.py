import numpy as np
from scipy.stats import rankdata

from colse.cdf_base import CDFBase
from colse.spline import monotone_cubic_spline


class SplineCDFModel(CDFBase):
    def __init__(self, **kwargs) -> None:
        np.random.seed(42)
        self.spline = None
        self.model_bytes = None
        self.no_of_rows = None
        self.sampled_distance = kwargs.get("sampled_distance", None)

    def fit(self, X, y=None):
        self.no_of_rows = X.shape[0]
        self.cum_sum = rankdata(X, method="min")
        arg_sort = np.argsort(self.cum_sum)

        x_squeeze = X.squeeze()
        x_unique = np.unique(x_squeeze[arg_sort])
        y_unique = np.unique(self.cum_sum[arg_sort])
        if self.sampled_distance is None:
            x = x_unique
            y = y_unique
        else:
            x = x_unique[:: self.sampled_distance]
            y = y_unique[:: self.sampled_distance]

        self.spline, _, self.model_bytes = monotone_cubic_spline(x, y)

        return self

    def predict(self, value):
        return max(self.spline(value) / self.no_of_rows, 0)

    def get_model_size(self):
        assert self.model_bytes is not None, "Model not fit yet"
        return self.model_bytes

    def get_range_cdf(self, lb, ub):
        if lb == -np.inf and ub == np.inf:
            return 1

        if lb == -np.inf:
            cdf_lb = 0
        else:
            cdf_lb = self.predict(lb)

        if ub == np.inf:
            cdf_ub = 1
        else:
            cdf_ub = self.predict(ub)

        return cdf_ub - cdf_lb / self.no_of_rows
