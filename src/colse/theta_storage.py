import json
import os
from multiprocessing import Pool
import time

from loguru import logger

from colse.copula_functions import get_theta


class ThetaStorage:

    def __init__(self, copula_type, no_of_columns, parellel=True):
        self.copula_type = copula_type
        self.no_of_columns = no_of_columns
        self.parellel = parellel

    def _calculate_theta(self, data_np):
        iterable = []
        ij_iterable = []
        for i in range(self.no_of_columns):
            for j in range(i + 1, self.no_of_columns):
                iterable.append((self.copula_type, data_np[i, :], data_np[j, :]))
                ij_iterable.append((i, j))
        if self.parellel:
            logger.info("Parellel Dequantization")
            with Pool() as pool:
                results = pool.map(get_theta, iterable)
        else:
            results = [get_theta(i) for i in iterable]

        theta_dict = {(i, j): val for val, (i, j) in zip(results, ij_iterable)}
        return theta_dict

    def get_theta(self, data_np, cache_name=None):
        if cache_name is not None and os.path.exists(cache_name):
            logger.info(f"Loading theta from cache: {cache_name}")
            return json.load(open(cache_name, "r"))
        start_time_theta_calc = time.perf_counter()
        theta_dict = self._calculate_theta(data_np)
        logger.info(f"Time Taken for Theta Calculation: {time.perf_counter() - start_time_theta_calc}")
        if cache_name is not None:
            json.dump(theta_dict, open(cache_name, "w"))
            logger.info(f"Saving theta to cache: {cache_name}")
        
        logger.info(f"Result Dict: {theta_dict}")
        return theta_dict
