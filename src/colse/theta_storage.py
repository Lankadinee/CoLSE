import os
import time
from multiprocessing import Pool

from loguru import logger
from rich.console import Console
from rich.table import Table

from colse.copula_functions import get_theta
from colse.pickle_utils import pickle_load, pickle_save


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
            logger.info("Parellel Theta Calculation")
            with Pool() as pool:
                results = pool.map(get_theta, iterable)
        else:
            results = [get_theta(i) for i in iterable]

        theta_dict = {(i, j): val for val, (i, j) in zip(results, ij_iterable)}

        return theta_dict

    def get_theta(self, data_np, cache_name=None):
        if cache_name is not None and os.path.exists(cache_name):
            logger.info(f"Loading theta from cache: {cache_name}")
            theta_dict = pickle_load(cache_name)
            self.show_theta_table(theta_dict)
            return theta_dict
        
        start_time_theta_calc = time.perf_counter()
        theta_dict = self._calculate_theta(data_np)
        logger.info(
            f"Time Taken for Theta Calculation: {time.perf_counter() - start_time_theta_calc}"
        )
        if cache_name is not None:
            pickle_save(theta_dict, cache_name)

        # logger.info(f"Result Dict: {theta_dict}")

        self.show_theta_table(theta_dict)
        logger.info("Theta Calculation Completed")
        # Return the theta dictionary for further use
        return theta_dict

    def show_theta_table(self, theta_dict):
        table_theta = Table(title="Theta Calculation")
        table_theta.add_column("Theta", justify="center", style="cyan", no_wrap=True)
        for i in range(self.no_of_columns):
            table_theta.add_column(f"Column {i + 1}")
        
        for j in range(self.no_of_columns):
            row = [f"Column {j + 1}"]
            for i in range(self.no_of_columns):
                if i == j:
                    row.append("N/A")
                else:
                    theta_value = theta_dict.get((i, j), theta_dict.get((j, i), "N/A"))
                    row.append(f"{theta_value:.4f}")
            table_theta.add_row(*row)
        console = Console()
        console.print(table_theta)