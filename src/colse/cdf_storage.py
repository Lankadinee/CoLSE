from pathlib import Path
import time
import pickle
from loguru import logger
import numpy as np
from tqdm import tqdm
from colse.cdf_dataframe import CDFDataFrame
from colse.emphirical_cdf import EMPMethod
from colse.data_path import get_data_path


class CDFStorage():
    """Calculate the CDF values for the given queries and store them in a file"""
    def __init__(self, cdf_model_cls, cached_name_string, override=False, *args, **kwargs):
        self.cdf_dataframe = CDFDataFrame(cdf_model_cls, *args, **kwargs)
        self.cache_name = get_data_path() / f"CDF_cache/{cached_name_string}.pkl"
        self.trained = False
        self.override = override
        self.data_split = kwargs.get("data_split", "NA")
        self.query_cache_name = get_data_path() / f"CDF_cache/{cached_name_string}_{self.data_split}.npy"


    def fit(self, df):
        start_time = time.time()
        self.trained = True
        if not self.cache_name.exists() or self.override:
            self.cdf_dataframe.fit(df)
            # save the cdf_dataframe in a file, pickle it
            logger.info(f"Saving the cdf_dataframe in {self.cache_name}")
            with open(self.cache_name, 'wb') as f:
                pickle.dump(self.cdf_dataframe, f)
        else:
            logger.info(f"Loading the cdf_dataframe from {self.cache_name}")
            with open(self.cache_name, 'rb') as f:
                self.cdf_dataframe = pickle.load(f)

        end_time = time.time()
        logger.info(f"Time taken to calculate CDFs: {end_time - start_time:.4f} seconds")
    
    def get_converted_cdf(self, X, column_indexes, nproc=4, cache=True):
        assert self.trained, "Model not trained yet"

        if cache and not self.override and self.query_cache_name.exists():
            logger.info(f"Loading the cdf values for queries from {self.query_cache_name}")
            return np.load(f"{self.query_cache_name}")
        
        x_shape = X.shape[1]
        if nproc <= 1:
            X_cdf = np.array([self.cdf_dataframe.predict(list(X[:, idx]), column_indexes[idx // 2]) for idx in range(x_shape)]).T
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # logger.info(f"Converting queries using {nproc} processors")
            with ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(self.predict, list(X[:, idx]), column_indexes[idx // 2]) for idx in range(x_shape)]
                # X_cdf =  np.array([f.result() for f in futures])

                results = []
                # for future in tqdm(as_completed(futures), total=len(futures)):
                for future in as_completed(futures):
                    results.append(future.result())
                X_cdf =  np.array([f.result() for f in futures]).T
                # TODO - chatGPT told that this order would not preserve, but from the tests seems like it did. So Im keeping it like this. 
        if cache:
            """save the cdf values, in a file"""
            logger.info(f"Saving the cdf values for queries in {self.query_cache_name}")
            np.save(f"{self.query_cache_name}", X_cdf)
        return X_cdf


if __name__ == "__main__":
    p1 = "/home/titan/phd/megadrive/query-optimization-methods/library/data/CDF_cache/power_test_data.npy"
    p2 = "/home/titan/phd/megadrive/query-optimization-methods/library/data/CDF_cache/power_test_datat.npy"

    """load p1 transpose and save it in p2"""
    data = np.load(p1)
    data = data.T

    np.save(p2, data)

    import pandas as pd
    from cest.cdf.emphirical_cdf import EmpiricalCDFModel

    df = pd.DataFrame(
        {
            "column1": [1, 2, 3, 4, 5] * 100,
            "column2": [5, 4, 3, 2, 1] * 100,
            "column3": [1, 2, 3, 4, 5] * 100,
        }
    )

    cdf_df = CDFStorage(EmpiricalCDFModel, cached_name_string='Power_test_data', override=False, emp_method=EMPMethod.RELATIVE)
    cdf_df.fit(df)

    print(cdf_df.get_model_size())

    # X = np.array([[1, 5, 1], [2, 4, 2], [3, 3, 3], [4, 2, 4], [5, 1, 5]])
    # print(cdf_df.get_converted_cdf(X, [0, 1, 2], nproc=2))
    