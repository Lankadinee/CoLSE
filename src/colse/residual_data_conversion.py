import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from colse.custom_data_generator import CustomDataGen
from colse.data_conversion_params import DataConversionParamValues, DataConversionParams
from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.spline_dequantizer import SplineDequantizer


@dataclass
class ResidualData:
    dataset_name: DatasetNames
    query: np.ndarray
    n_query: np.ndarray
    y_bar: np.ndarray
    gt: np.ndarray
    q_error: np.ndarray
    x_cdf: np.ndarray
    no_of_rows: int
    min_values: np.ndarray
    max_values: np.ndarray

    def __str__(self):
        return f"ResidualData - for dataset: {self.dataset_name} NoOfRows:{self.no_of_rows}"


class DataConversion:
    VERSION = "1-0-0"

    def __init__(self, dataset_name: DatasetNames, df_param_obj: DataConversionParamValues):
        self.dataset_name = dataset_name
        self.max_values = df_param_obj.max_values
        self.min_values = df_param_obj.min_values
        self.no_of_rows = df_param_obj.no_of_rows

    def convert(self, excel_file_path, use_cache=True):
        # Create ReData folder if it doesn't exist
        resdata_folder = get_data_path() / "ResData"
        resdata_folder.mkdir(parents=True, exist_ok=True)
        if isinstance(excel_file_path, str):
            excel_file_path = Path(excel_file_path)
        name = excel_file_path.stem
        cache_name = (
            resdata_folder / f"{self.dataset_name}_CV-{self.VERSION}_{name}.pkl"
        )
        cahche_path = get_data_path() / cache_name
        if use_cache and cahche_path.exists():
            logger.info(f"Using cached data from {cahche_path}")
            with open(cahche_path, "rb") as f:
                return pickle.load(f)

        start_time = time.time()
        logger.info(f"Converting data Started..., using {name}")
        s_dequantize = SplineDequantizer(dataset_type=self.dataset_name)
        sd_file_name = s_dequantize.get_dequantized_dataset_name()
        if sd_file_name is None:
            sd_file_name = self.dataset_name.get_file_path()
        if self.max_values is None:
            raise ValueError("Max values are not set")

        logger.info(f"Loading data from {excel_file_path}")
        df = pd.read_excel(excel_file_path)

        df = df[df['X'].apply(lambda x: isinstance(x, str))]
        logger.info(f"Loaded {len(df)} rows")
        x_cdf = df["X"].to_list()
        x_cdf = [np.array(xc.split(","), dtype=np.float64).tolist() for xc in x_cdf]
        query = df["mapped_query"].to_numpy()
        # y_bar = np.log2(df["y_bar"].to_numpy() * self.no_of_rows + 1)
        gt = df["gt"].to_numpy()
        y = np.log2(gt + 1)
        q_error = df["q_error"].to_numpy()
        diff = [
            (self.max_values[i] - self.min_values[i])
            for i in range(len(self.max_values))
        ]

        normalized_query = []
        for q in tqdm(query):
            q_np = np.array(q.split(","), dtype=np.float64).tolist()
            norm_q = np.array(
                [
                    (val - self.min_values[int(i // 2)]) / diff[int(i // 2)]
                    for i, val in enumerate(q_np)
                ]
            )
            norm_q[norm_q == -np.inf] = 0
            norm_q[norm_q == np.inf] = 1
            normalized_query.append(norm_q)

        """concatenate normalized_query and y_bar"""
        # x = np.concatenate((normalized_query, y_bar.reshape(-1, 1)), axis=1)
        n_query = np.array(normalized_query)
        # y_bar  = df["y_bar"].to_numpy()   # TODO - Strange behavior - check later
        y_bar = np.array(df["y_bar"].to_list())
        res_data = ResidualData(
            dataset_name=self.dataset_name,
            query=query,
            n_query=n_query,
            y_bar=np.clip(y_bar, 0, 1),
            gt=gt,
            q_error=q_error,
            x_cdf=x_cdf,
            no_of_rows=self.no_of_rows,
            min_values=self.min_values,
            max_values=self.max_values,
        )

        with open(cahche_path, "wb") as f:
            pickle.dump(res_data, f)

        logger.info(f"Data conversion done in {time.time() - start_time} seconds")
        return res_data


if __name__ == "__main__":
    excel_path = "/home/titan/phd/megadrive/query-optimization-methods/experiment_15/dvine_copula_dynamic_recursive tests/results/dynamic_compare_results.xlsx"
    dc = DataConversion()
    rd = dc.convert(excel_path, use_cache=False)
    print(rd)
