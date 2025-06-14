import json
import random
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.datasets.dataset_census import generate_dataset as generate_dataset_census
from colse.datasets.dataset_census import get_queries_census
from colse.datasets.dataset_correlated_02 import (
    generate_dataset_correlated_02,
    get_queries_correlated_02,
)
from colse.datasets.dataset_correlated_04 import (
    generate_dataset_correlated_04,
    get_queries_correlated_04,
)
from colse.datasets.dataset_correlated_06 import (
    generate_dataset_correlated_06,
    get_queries_correlated_06,
)
from colse.datasets.dataset_correlated_08 import (
    generate_dataset_correlated_08,
    get_queries_correlated_08,
)
from colse.datasets.dataset_dmv import generate_dataset as generate_dataset_dmv
from colse.datasets.dataset_dmv import get_queries_dmv
from colse.datasets.dataset_forest import generate_dataset as generate_dataset_forest
from colse.datasets.dataset_forest import get_queries_forest
from colse.datasets.dataset_gas import generate_dataset as generate_dataset_gas
from colse.datasets.dataset_gas import get_queries
from colse.datasets.dataset_power import generate_dataset as generate_dataset_power
from colse.datasets.dataset_power import get_queries_power
from colse.datasets.dataset_tpch_lineitem import (
    generate_dataset_tpch_lineitem_10,
    generate_dataset_tpch_lineitem_20,
    generate_dataset_tpch_sf2_z0_lineitem,
    generate_dataset_tpch_sf2_z1_lineitem,
    generate_dataset_tpch_sf2_z2_lineitem,
    generate_dataset_tpch_sf2_z3_lineitem,
    generate_dataset_tpch_sf2_z4_lineitem,
    get_queries_tpch_lineitem_10,
    get_queries_tpch_lineitem_20,
    get_queries_tpch_sf2_z0_lineitem,
    get_queries_tpch_sf2_z1_lineitem,
    get_queries_tpch_sf2_z2_lineitem,
    get_queries_tpch_sf2_z3_lineitem,
    get_queries_tpch_sf2_z4_lineitem,
)
from colse.datasets.dataset_v1 import generate_dataset as generate_dataset_v1
from colse.datasets.dataset_v2 import generate_dataset as generate_dataset_v2
from colse.datasets.dataset_v3 import generate_dataset as generate_dataset_v3
from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX
from colse.datasets.query_filter import filter_queries
from colse.datasets.variable_dataset import (
    generate_dataset as generate_dataset_variable,
)
from colse.transform_datasets import convert_df_to_dequantize

DS_TYPE_MAPPER = {
    DatasetNames.FOREST_DATA: generate_dataset_forest,
    DatasetNames.POWER_DATA: generate_dataset_power,
    DatasetNames.CENSUS_DATA: generate_dataset_census,
    DatasetNames.DMV_DATA: generate_dataset_dmv,
    DatasetNames.TPCH_SF2_Z1_LINEITEM: generate_dataset_tpch_sf2_z1_lineitem,
    DatasetNames.TPCH_SF2_Z2_LINEITEM: generate_dataset_tpch_sf2_z2_lineitem,
    DatasetNames.TPCH_SF2_Z3_LINEITEM: generate_dataset_tpch_sf2_z3_lineitem,
    DatasetNames.TPCH_SF2_Z4_LINEITEM: generate_dataset_tpch_sf2_z4_lineitem,
    DatasetNames.TPCH_LINEITEM_10: generate_dataset_tpch_lineitem_10,
    DatasetNames.TPCH_LINEITEM_20: generate_dataset_tpch_lineitem_20,
    DatasetNames.CORRELATED_02: generate_dataset_correlated_02,
    DatasetNames.CORRELATED_04: generate_dataset_correlated_04,
    DatasetNames.CORRELATED_06: generate_dataset_correlated_06,
    DatasetNames.CORRELATED_08: generate_dataset_correlated_08,
}

QUERY_GENERATE_MAPPER = {
    DatasetNames.FOREST_DATA: get_queries_forest,
    DatasetNames.POWER_DATA: get_queries_power,
    DatasetNames.CENSUS_DATA: get_queries_census,
    DatasetNames.DMV_DATA: get_queries_dmv,
    DatasetNames.TPCH_SF2_Z1_LINEITEM: get_queries_tpch_sf2_z1_lineitem,
    DatasetNames.TPCH_SF2_Z2_LINEITEM: get_queries_tpch_sf2_z2_lineitem,
    DatasetNames.TPCH_SF2_Z3_LINEITEM: get_queries_tpch_sf2_z3_lineitem,
    DatasetNames.TPCH_SF2_Z4_LINEITEM: get_queries_tpch_sf2_z4_lineitem,
    DatasetNames.TPCH_LINEITEM_10: get_queries_tpch_lineitem_10,
    DatasetNames.TPCH_LINEITEM_20: get_queries_tpch_lineitem_20,
    DatasetNames.CORRELATED_02: get_queries_correlated_02,
    DatasetNames.CORRELATED_04: get_queries_correlated_04,
    DatasetNames.CORRELATED_06: get_queries_correlated_06,
    DatasetNames.CORRELATED_08: get_queries_correlated_08,
}

def none_class():
    return None


SCALAR_MAPPER = {
    "min_max": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "none": none_class,
}


@dataclass
class MoreInfo:
    min_query_value: float = None
    max_query_value: float = None


class CustomDataGen:
    LOAD_FROM_CACHE = False

    def __init__(
        self,
        no_of_rows: Optional[int] = 500000,
        no_of_queries: Optional[int] = 1000,
        dataset_type: DatasetNames = DatasetNames.FOREST_DATA,
        data_file_name: str | None = None,
        query_file_name: str | None = None,
        data_split="test",
        selected_cols: Optional[list] = None,
        scalar_type: str = "min_max",
        dequantize: bool = False,
        seed: int = 0,
        is_range_queries: bool = True,
        verbose: bool = True,
        **kwargs,
    ):

        np.random.seed(seed)
        self.seed = seed
        self.selected_cols = selected_cols
        col_str = (
            "".join([str(cid) for cid in selected_cols])
            if selected_cols is not None
            else "all"
        )
        self.datagen_name = (
            get_data_path()
            / f"CDG_cache/custom_data_gen_{dataset_type}_R{no_of_rows}_S{no_of_queries}_C{col_str}_S{scalar_type}_split-{data_split}_q{dequantize}"
        )

        self.dataset_type = dataset_type
        self.data_split = data_split
        self.no_of_queries = no_of_queries
        self.no_of_rows = no_of_rows
        self.no_of_features = None
        self.query_l = None
        self.query_r = None
        self.true_card = None
        self.scalar_type = scalar_type
        self.scaler = None
        self.sample_query_col_list = []
        self.dequantize = dequantize
        self.enable_query_dequantize = kwargs.get("enable_query_dequantize", False)
        self.verbose = verbose
        self.is_range_queries = is_range_queries
        self.more_info = MoreInfo()
        self.splines = dict()
        self.data_file_name = data_file_name
        self.query_file_name = query_file_name
        (
            logger.info(
                f"Checking saved data generator {self.datagen_name.name} exists {self.datagen_name.exists()}"
            )
            if CustomDataGen.LOAD_FROM_CACHE
            else ""
        )
        if CustomDataGen.LOAD_FROM_CACHE and self.datagen_name.exists():
            with open(self.datagen_name / "other_data.json", "r") as f:
                other_data = json.load(f)
                assert other_data["dataset_type"] == self.dataset_type
                assert self.is_range_queries == other_data["is_range_queries"]
                assert self.seed == other_data["seed"]
                self.no_of_queries = other_data["sample_count"]
                self.no_of_rows = other_data["no_of_rows"]

            df = pd.read_csv(self.datagen_name / "df.csv")
            self.query_l = np.load(self.datagen_name / "query_l.npy")
            self.query_r = np.load(self.datagen_name / "query_r.npy")
            self.true_card = np.load(self.datagen_name / "true_card.npy")
            self.df = self.generate_dataset(df=df)
            logger.info(f"Loaded custom data generator from {self.datagen_name.name}")
        else:
            self.df = self.generate_dataset()
            self.no_of_queries = self.generate_queries(is_range_queries)
            self.save(self.datagen_name) if CustomDataGen.LOAD_FROM_CACHE else None

        logger.info(
            f"Dataset {self.dataset_type} generated with {self.no_of_rows} rows and {self.no_of_queries} samples"
        )

    def filter_queries_by_cols(self, columns: List[int]):
        query_l = self.query_l[columns]
        query_r = self.query_r[columns]
        true_card = self.true_card

    def save(self, dir_path: Path):
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        self.df.to_csv(dir_path / "df.csv", index=False)

        """save numpy arrays"""
        np.save(dir_path / "query_l.npy", self.query_l)
        np.save(dir_path / "query_r.npy", self.query_r)
        np.save(dir_path / "true_card.npy", self.true_card)

        other_data = {
            "no_of_rows": self.no_of_rows,
            "no_of_features": self.no_of_features,
            "sample_count": self.no_of_queries,
            "dataset_type": self.dataset_type,
            "is_range_queries": self.is_range_queries,
            "seed": self.seed,
            "no_of_cols": self.selected_cols,
        }
        with open(dir_path / "other_data.json", "w") as f:
            json.dump(other_data, f)

        logger.info(f"Saved custom data generator to {dir_path}")
        return True

    def generate_dataset(self, seed=0, df=None):
        logger.info(f"Generating dataset {self.dataset_type}")
        if self.dataset_type not in DS_TYPE_MAPPER:
            raise ValueError(f"Dataset type {self.dataset_type} not supported")

        if df is None:
            df = DS_TYPE_MAPPER[self.dataset_type](
                no_of_rows=self.no_of_rows,
                selected_cols=self.selected_cols,
                data_file_name=self.data_file_name,
            )

            if self.dequantize:
                df, self.splines = convert_df_to_dequantize(df)

        if self.no_of_rows is None:
            self.no_of_rows = df.shape[0]

        self.scaler = SCALAR_MAPPER[self.scalar_type]()

        """replace the inf values with 1"""
        if self.scaler is not None:
            df_copy = df.copy()
            df_copy = df_copy.replace([np.inf, -np.inf], 1)
            self.scaler.fit(df_copy.values)

        self.no_of_features = df.shape[1]
        return df

    def actual_cdf(self, lb, ub=None):
        X = self.df.to_numpy().transpose()
        value = 1
        index = 0

        if ub is None:
            for x1 in lb:
                value *= X[index] <= x1
                index += 1
        else:
            for x1, x2 in zip(lb, ub):
                value *= (X[index] >= x1) * (X[index] <= x2)
                index += 1

        return value.sum()

    def generate_queries(self, is_range):
        if is_range:
            ret = self.generate_range_queries()
        else:
            ret = self.generate_point_queries()

        return ret

    def generate_point_queries(self):
        logger.info(f"Generating queries for dataset {self.dataset_type}")
        self.query_r = self.s_df.to_numpy()
        self.true_card = np.array([self.actual_cdf(lb) for lb in self.query_r])
        return self.no_of_queries

    def _get_actual_cardinality(self):
        logger.info("Generating true cardinality") if self.verbose else None
        actual_card = []
        for ub, lb in tqdm(
            zip(self.query_r, self.query_l), total=self.query_l.shape[0]
        ):
            actual_cdf_value = self.actual_cdf(lb, ub)
            actual_card.append(actual_cdf_value if actual_cdf_value > 1 else 1)
        return np.array(actual_card)

    def generate_range_queries(self):
        logger.info(f"Generating queries for dataset {self.dataset_type}")
        if self.dataset_type not in QUERY_GENERATE_MAPPER:
            if self.no_of_queries is None:
                self.no_of_queries = int(self.df.shape[0] // 10)
                logger.info(
                    f"Sample count not provided. Setting it to {self.no_of_queries}"
                )

            sample_data = (
                self.df.sample(n=self.no_of_queries * 2).to_numpy().transpose()
            )
            rng = np.random.default_rng()

            def shuffle_and_return(x):
                rng.shuffle(x)
                return x

            sample_data = np.array([shuffle_and_return(d) for d in sample_data])
            data_samples = np.array(
                [
                    [d[: self.no_of_queries], d[self.no_of_queries :]]
                    for d in sample_data
                ]
            )
            data_samples = np.sort(data_samples, axis=1)
            data_samples = data_samples.transpose(1, 2, 0)
            self.query_l, self.query_r = data_samples[0], data_samples[1]
            self.true_card = self._get_actual_cardinality()
            return self.no_of_queries
        elif self.sample_query_col_list is None or len(self.sample_query_col_list) > 0:
            no_inf = True
            if self.sample_query_col_list is None:
                self.sample_query_col_list = list(range(self.no_of_features))
                no_inf = True

            # self.query_l, self.query_r = QUERY_GENERATE_MAPPER_CUSTOM[
            #     self.dataset_type
            # ](
            #     no_of_queries=self.no_of_queries,
            #     queried_columns=self.sample_query_col_list,
            #     no_inf=no_inf,
            # )
            self.true_card = self._get_actual_cardinality()
            return self.query_l.shape[0]
        else:
            self.query_l, self.query_r, self.true_card = QUERY_GENERATE_MAPPER[
                self.dataset_type
            ](
                no_of_queries=self.no_of_queries,
                data_split=self.data_split,
                min_value=self.no_of_rows / 1000,
                enable_query_dequantize=self.enable_query_dequantize,  # Mapping queried values to dequantized values
                query_file_name=self.query_file_name,
            )
            if self.selected_cols is not None:
                no_of_cols = self.query_l.shape[1]
                remove_cols = [
                    i for i in range(no_of_cols) if i not in self.selected_cols
                ]
                self.query_l, self.query_r = filter_queries(
                    self.query_l, self.query_r, remove_cols
                )
                self.true_card = self._get_actual_cardinality()

            return self.query_l.shape[0]

    def get_groups(self, no_of_features=3):
        groups = [
            [1, 2, 3],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [1, 2, 12],
            [5, 6, 7],
            [8, 9, 10],
            [1, 12, 14],
        ]
        # how to substract 1 from all elements in the list
        groups = [[g - 1 for g in group] for group in groups]
        return groups

    """Get random feature combinations from a given list of numbers"""

    def get_random_feature_combinations(
        self, no_of_features=3, length=None, no_of_combinations=None
    ):
        number_list = list(range(1, no_of_features + 1))
        length = len(number_list) if length is None else length
        # Generate all possible combinations of the specified length
        all_combinations = list(combinations(number_list, length))
        return (
            all_combinations
            if no_of_combinations is None
            else random.sample(all_combinations, no_of_combinations)
        )

    def get_permutations_per_groups(
        self, input_list, length=None, no_of_permutations=None
    ):
        length = len(input_list) if length is None else length
        permutations_list = list(permutations(input_list, length))
        return (
            permutations_list
            if no_of_permutations is None
            else random.sample(permutations_list, no_of_permutations)
        )

    def get_df(self):
        return self.df

    def get_values(self, group=None):
        return (
            np.array([self.df[f"{ROW_PREFIX}{att}"] for att in group])
            if group is not None
            else self.df.to_numpy().transpose()
        )

    def get_sample_values(self, group):
        return np.array([self.s_df[f"{SAMPLE_PREFIX}{att}"] for att in group])


if __name__ == "__main__":
    cd_obj = CustomDataGen(
        no_of_rows=None, no_of_queries=None, dataset_type=DatasetNames.FOREST_DATA
    )
    logger.info("Generating range queries")
    logger.info(f"Sample count: {cd_obj.query_l.shape[0]} | {cd_obj.no_of_queries}")
    # print(cd_obj.query_l.shape)
    # for lb, ub, card in zip(cd_obj.query_l, cd_obj.query_r, cd_obj.true_card):
    #     # print(lb, ub, card)
    #     actual = cd_obj.actual_cdf(lb, ub)
    #     print(actual, card)
    #     assert actual == card
    # groups = cd_obj.get_values(group=[1, 2, 3])
    # print(groups.shape)
    # sample_data = cd_obj.get_sample_values(group=[1, 2, 3])
    # print(sample_data.shape)
