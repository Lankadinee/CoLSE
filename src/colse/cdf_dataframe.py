
from loguru import logger
import numpy as np
from numpy import isin
from tqdm import tqdm

from colse.optimized_emp_cdf import OptimizedEmpiricalCDFModel

from colse.emphirical_cdf import EmpiricalCDFModel
from colse.spline_cdf import SplineCDFModel

class CDFDataFrame:
    """For each column in the dataframe, a CDF model is created and stored in a dictionary"""
    def __init__(self, cdf_model_cls, **kwargs):
        self.cdf_model_cls = cdf_model_cls
        self.models = {}
        self.columns = []
        
        self.kwargs = kwargs

    def get_model_size(self, rtype="string"):
        valid_instances = [OptimizedEmpiricalCDFModel, EmpiricalCDFModel, SplineCDFModel]
        if self.cdf_model_cls not in valid_instances:
            raise ValueError(f"Model class should be one of {valid_instances}")
        size_in_bytes = sum([model.get_model_size() for model in self.models.values() ])
        if rtype == "kb":
            return size_in_bytes / 1024
        
        if size_in_bytes < 1024:
            return f"{size_in_bytes} bytes"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.3f} KB"
        else:
            return f"{size_in_bytes / 1024 / 1024:.3f} MB"

    def fit(self, df, nproc=1):
        self.columns = df.columns
        logger.info(f"CDF dataframe fitting for columns: {self.columns}")

        if nproc <= 1:
            loop = tqdm(self.columns)
            for col in loop:
                loop.set_description(f"Fitting {col}")
                self.models[col] = self.cdf_model_cls(verbose=False, **self.kwargs)
                self.models[col].fit(df[col].to_numpy())
        else:
            raise NotImplemented
            from concurrent.futures import ProcessPoolExecutor, as_completed
            def model_fit(col_name):
                self.models[col_name] = self.cdf_model_cls(**self.kwargs)
                self.models[col_name].fit(df[col_name].to_numpy())

            with ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(model_fit, col_name) for col_name in self.columns]
                results =  [f.result() for f in futures]
                #  TODO -  AttributeError: Can't pickle local object 'CDFDataFrame.fit.<locals>.model_fit'
                # Use tqdm to show progress as futures complete
                # for future in tqdm(as_completed(futures), total=len(futures)):
                #     results.append(future.result())

    def _get_cdf(self, value, column_name):
        if value == -np.inf:
            return 0
        elif value == np.inf:
            return 1
        else:
            return self.models[column_name].predict(value)
    
    def get_unique_value_count(self, col):
        return self.models[col].get_unique_value_count()

    def predict(self, value, col=None):
        if isinstance(value, list):
            if len(value) == len(self.columns):
                return np.array([self._get_cdf(val, self.columns[idx]) for idx, val in enumerate(value)])
            elif isinstance(col, str):
                assert col in self.columns, f"Column {col} not found in the dataframe"
                return np.array([self._get_cdf(val, col) for val in value])
            elif isinstance(col, int):
                return np.array([self._get_cdf(val, self.columns[col]) for val in value])
            else:
                raise ValueError("Number of values should be equal to the number of columns")
        else:
            assert col is not None, "Column name or index is required"
            if isinstance(col, int):
                return self._get_cdf(value, self.columns[col])
            elif isin(col, self.columns):
                return self._get_cdf(value, col)


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame(
        {
            "column1": [1, 2, 3, 4, 5] * 5,
            "column2": [5, 4, 3, 2, 1] * 5,
            "column3": [1, 2, 3, 4, 5] * 5,
        }
    )

    cdf_df = CDFDataFrame(EmpiricalCDFModel)
    cdf_df.fit(df, nproc=1)
    # print(cdf_df.predict([1, 2, 3], 0))
    print(cdf_df.predict([1, 2, 3], "column2"))

    print("Model size: ", cdf_df.get_model_size())
    # print(cdf_df.predict(1, "column1"))
    # print(cdf_df.predict(1, 0))
