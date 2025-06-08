from enum import Enum

from colse.data_path import get_data_path


class DatasetNames(str, Enum):
    CUSTOM = "custom"
    PAYROLL_DATA = "payroll_data"
    GAS_DATA = "gas_data"
    CUSTOM_2 = "custom_2"
    FOREST_DATA = "forest"
    POWER_DATA = "power"
    VARIABLE = "variable"
    CENSUS_DATA = "census"
    DMV_DATA = "dmv"
    TPCH_SF2_Z0_LINEITEM = "tpch_sf2_z0_lineitem"
    TPCH_SF2_Z1_LINEITEM = "tpch_sf2_z1_lineitem"
    TPCH_SF2_Z2_LINEITEM = "tpch_sf2_z2_lineitem"
    TPCH_SF2_Z3_LINEITEM = "tpch_sf2_z3_lineitem"
    TPCH_SF2_Z4_LINEITEM = "tpch_sf2_z4_lineitem"
    CORRELATED_2 = "synthetic_correlated_2"
    CORRELATED_3 = "synthetic_correlated_3"
    CORRELATED_4 = "synthetic_correlated_4"
    CORRELATED_6 = "synthetic_correlated_6"
    CORRELATED_8 = "synthetic_correlated_8"
    CORRELATED_10 = "synthetic_correlated_10"
    TPCH_LINEITEM_10 = "tpch_lineitem_10"
    TPCH_LINEITEM_20 = "tpch_lineitem_20"
    CORRELATED_02 = "correlated_02"
    CORRELATED_04 = "correlated_04"
    CORRELATED_06 = "correlated_06"
    CORRELATED_08 = "correlated_08"

    def __str__(self):
        return self.value

    def get_file_path(self):
        if self == DatasetNames.POWER_DATA:
            return get_data_path(self.value) / "original.csv"
        elif self == DatasetNames.DMV_DATA:
            return get_data_path(self.value) / "dmv.csv"
        elif self == DatasetNames.FOREST_DATA:
            return get_data_path(self.value) / "forest.csv"
        else:
            raise ValueError(f"Dataset {self} not supported")

    def get_non_continuous_columns(self):
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
                "Revocation_Indicator"
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
            or self == DatasetNames.TPCH_SF2_Z0_LINEITEM
        ):
            return 15
        elif self == DatasetNames.TPCH_SF2_Z2_LINEITEM:
            return 15
        elif self == DatasetNames.TPCH_SF2_Z3_LINEITEM:
            return 15
        elif self == DatasetNames.TPCH_SF2_Z4_LINEITEM:
            return 15
        elif self == DatasetNames.CORRELATED_2:
            return 2
        elif self == DatasetNames.CORRELATED_3:
            return 3
        elif self == DatasetNames.CORRELATED_4:
            return 4
        elif self == DatasetNames.CORRELATED_6:
            return 6
        elif self == DatasetNames.CORRELATED_8:
            return 8
        elif (
            self == DatasetNames.CORRELATED_10
            or self == DatasetNames.CORRELATED_02
            or self == DatasetNames.CORRELATED_04
            or self == DatasetNames.CORRELATED_06
            or self == DatasetNames.CORRELATED_08
        ):
            return 10
        elif self == DatasetNames.TPCH_LINEITEM_10:
            return 15
        elif self == DatasetNames.TPCH_LINEITEM_20:
            return 15
        else:
            raise ValueError(f"Dataset {self} not supported")
