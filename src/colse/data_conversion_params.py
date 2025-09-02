import pickle

from loguru import logger
from colse.custom_data_generator import CustomDataGen
from colse.data_path import DataPathDir, get_data_path
from colse.dataset_names import DatasetNames

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataConversionParamValues:
    min_values: np.ndarray
    max_values: np.ndarray
    no_of_rows: int


class DataConversionParams:
    def __init__(self, dataset_name: DatasetNames, data_update_type: str = None):

        self._dataset_name = dataset_name
        self._data_update_type = data_update_type
        print("Data set name type: ", type(self._dataset_name))

        _file_name = f"{self._data_update_type}.pkl" if self._data_update_type else f"{self._dataset_name.value}_original.pkl"
        self._cache_name = get_data_path(DataPathDir.DATA_CONVERSION_PARAMS,  self._dataset_name.value) / _file_name
        logger.info(f"Data conversion params initialized for {self._dataset_name} with update type {self._data_update_type}")

    def store_data_conversion_params(self, dataset : CustomDataGen= None):

        dc_params = DataConversionParamValues(
            min_values=dataset.scaler.data_min_,
            max_values=dataset.scaler.data_max_,
            no_of_rows=dataset.no_of_rows
        )
        
        # save to pickle
        with open(self._cache_name, "wb") as f:
            pickle.dump(dc_params, f)
        logger.info(f"Data conversion params stored in {self._cache_name}")
    
    def store_data_conversion_params_df(self, dataset : pd.DataFrame):
        dc_params = DataConversionParamValues(
            min_values=dataset.min().to_numpy(),
            max_values=dataset.max().to_numpy(),
            no_of_rows=dataset.shape[0]
        )
        
        # save to pickle
        with open(self._cache_name, "wb") as f:
            pickle.dump(dc_params, f)
        logger.info(f"Data conversion params stored in {self._cache_name}")


    def load_data_conversion_params(self):
        if not self._cache_name.exists():
            raise FileNotFoundError(f"Data conversion params file not found: {self._cache_name}")
        
        with open(self._cache_name, "rb") as f:
            dc_params = pickle.load(f)
        logger.info(f"Data conversion params loaded from {self._cache_name}")
        return dc_params