from enum import Enum

from loguru import logger

from colse.data_path import get_data_path


class DatasetNames(str, Enum):
    FOREST_DATA = "forest"
    POWER_DATA = "power"
    CENSUS_DATA = "census"
    DMV_DATA = "dmv"
    TPCH_SF2_Z1_LINEITEM = "tpch_sf2_z1_lineitem"
    TPCH_SF2_Z2_LINEITEM = "tpch_sf2_z2_lineitem"
    TPCH_SF2_Z3_LINEITEM = "tpch_sf2_z3_lineitem"
    TPCH_SF2_Z4_LINEITEM = "tpch_sf2_z4_lineitem"
    TPCH_LINEITEM_10 = "tpch_lineitem_10"
    TPCH_LINEITEM_20 = "tpch_lineitem_20"
    CORRELATED_02 = "correlated_02"
    CORRELATED_04 = "correlated_04"
    CORRELATED_06 = "correlated_06"
    CORRELATED_08 = "correlated_08"
    IMDB_DATA = "imdb"
    CUSTOM_JOIN_DATA = "custom_join"

    def __str__(self):
        return self.value

    def is_tpch_type(self):
        return (
            self == DatasetNames.TPCH_SF2_Z1_LINEITEM
            or self == DatasetNames.TPCH_SF2_Z2_LINEITEM
            or self == DatasetNames.TPCH_SF2_Z3_LINEITEM
            or self == DatasetNames.TPCH_SF2_Z4_LINEITEM
            or self == DatasetNames.TPCH_LINEITEM_10
            or self == DatasetNames.TPCH_LINEITEM_20
        )
    
    def is_correlated_type(self):
        return (
            self == DatasetNames.CORRELATED_02
            or self == DatasetNames.CORRELATED_04
            or self == DatasetNames.CORRELATED_06
            or self == DatasetNames.CORRELATED_08
        )

    def get_file_path(self, filename=None, exist_check=True, pp_enb=True):
        if self == DatasetNames.POWER_DATA:
            dp = get_data_path(self.value) / (filename if filename else "original.csv")
        elif self == DatasetNames.DMV_DATA:
            dp = get_data_path(self.value) / (filename if filename else "dmv.parquet")
        elif self == DatasetNames.FOREST_DATA:
            dp = get_data_path(self.value) / (filename if filename else "forest.csv")
        elif self == DatasetNames.IMDB_DATA:
            dp = get_data_path(self.value) / (filename if filename else "samples_data.csv")
        elif self == DatasetNames.CUSTOM_JOIN_DATA:
            dp = get_data_path(self.value) / (filename if filename else "custom_join_dataset.xlsx")
        elif self == DatasetNames.CENSUS_DATA:
            dp = get_data_path(self.value) / (filename if filename else "census.csv")
        elif self.is_tpch_type():
            default_filename = "original_preprocessed.parquet" if pp_enb else "original.parquet"
            logger.info(f"Using {default_filename} for {self}")
            dp = get_data_path(self.value) / (
                filename if filename else default_filename
            )
        elif self.is_correlated_type():
            dp = get_data_path(self.value) / (
                filename if filename else "original.parquet"
            )
        else:
            raise ValueError(f"Dataset {self} not supported")
        if not dp.exists() and exist_check:
            raise FileNotFoundError(f"File {dp} not found")

        return dp

    def get_non_continuous_columns(self, **kwargs):
        if self == DatasetNames.FOREST_DATA:
            return []
        elif self == DatasetNames.POWER_DATA:
            return []
        elif self == DatasetNames.DMV_DATA:
            return [
                "Record_Type",
                "Registration_Class",
                "State",
                "County",
                "Body_Type",
                "Fuel_Type",
                "Color",
                "Scofflaw_Indicator",
                "Suspension_Indicator",
                "Revocation_Indicator",
            ]
        elif self == DatasetNames.CENSUS_DATA:
            return [
                "workclass",
                "education",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native_country",
            ]
        elif self.is_tpch_type():
            return [
                "l_returnflag",
                "l_linestatus",
                "l_shipinstruct",
                "l_shipmode",
            ]
        elif self.is_correlated_type():
            return []
        elif self == DatasetNames.CUSTOM_JOIN_DATA:
            if "table_name" in kwargs:
                columns_map = {
                    "customers": ["name", "city"],
                    "orders": [],
                    "products": ["product_name"],
                }
                return columns_map[kwargs["table_name"]]
            else:
                return []
        elif self == DatasetNames.IMDB_DATA:
            if "table_name" in kwargs:
                columns_map = {
                    "cast_info": ["role_id"],
                    "movie_companies": ["company_id", "company_type_id"],
                    "movie_info": ["info_type_id"],
                    "movie_keyword": ["keyword_id"],
                    "title": ["kind_id"],
                    "movie_info_idx": ["info_type_id"],
                }
                return columns_map[kwargs["table_name"]]
                # return []
            else:
                return [
                    "cast_info:role_id",
                    "movie_companies:company_id",
                    "movie_companies:company_type_id",
                    "movie_info:info_type_id",
                    "movie_keyword:keyword_id",
                    "title:kind_id",
                    "movie_info_idx:info_type_id",  
                ]
        else:
            raise ValueError(f"Dataset {self} not supported")

    def get_no_of_columns(self):
        if self == DatasetNames.FOREST_DATA:
            return 10
        elif self == DatasetNames.POWER_DATA:
            return 7
        elif self == DatasetNames.CENSUS_DATA:
            return 13
        elif self == DatasetNames.DMV_DATA:
            return 11
        elif (
            self == DatasetNames.TPCH_SF2_Z1_LINEITEM
        ):
            return 15
        elif self == DatasetNames.TPCH_SF2_Z2_LINEITEM:
            return 15
        elif self == DatasetNames.TPCH_SF2_Z3_LINEITEM:
            return 15
        elif self == DatasetNames.TPCH_SF2_Z4_LINEITEM:
            return 15
        elif (
            self == DatasetNames.CORRELATED_02
            or self == DatasetNames.CORRELATED_04
            or self == DatasetNames.CORRELATED_06
            or self == DatasetNames.CORRELATED_08
        ):
            return 10
        elif self == DatasetNames.TPCH_LINEITEM_10:
            return 15
        elif self == DatasetNames.TPCH_LINEITEM_20:
            return 15
        elif self == DatasetNames.IMDB_DATA or self == DatasetNames.CUSTOM_JOIN_DATA:
            raise ValueError(f"Dataset {self} not supported to get no of columns")
        else:
            raise ValueError(f"Dataset {self} not supported")

    def is_join_type(self):
        return self == DatasetNames.IMDB_DATA or self == DatasetNames.CUSTOM_JOIN_DATA

    def get_join_tables(self):
        if self == DatasetNames.IMDB_DATA:
            return [
                "title",
                "cast_info",
                "movie_companies",
                "movie_info",
                "movie_info_idx",
                "movie_keyword",
            ]
        elif self == DatasetNames.CUSTOM_JOIN_DATA:
            return [
                "customers",
                "orders",
                "products",
            ]
        else:
            raise ValueError(f"Dataset {self} not supported")