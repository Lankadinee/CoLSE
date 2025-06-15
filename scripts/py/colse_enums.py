


from enum import StrEnum, auto


def get_common_database_name(dataset_name):
    if "forest" in dataset_name:
        return "forest"
    elif "power" in dataset_name:
        return "power"
    elif "census" in dataset_name:
        return "census"
    elif "dmv" in dataset_name:
        return "dmv"
    elif "tpch" in dataset_name:
        return "tpch"
    elif "correlated_02" in dataset_name:
        return "correlated_02"
    elif "correlated_04" in dataset_name:
        return "correlated_04"
    elif "correlated_06" in dataset_name:
        return "correlated_06"
    elif "correlated_08" in dataset_name:
        return "correlated_08"
    elif "tpch_lineitem_10" in dataset_name:
        return "tpch_lineitem_10"
    elif "tpch_lineitem_20" in dataset_name:
        return "tpch_lineitem_20"
    elif "tpch_sf2_z1_lineitem" in dataset_name:
        return "tpch_sf2_z1_lineitem"
    elif "tpch_sf2_z2_lineitem" in dataset_name:
        return "tpch_sf2_z2_lineitem"
    elif "tpch_sf2_z3_lineitem" in dataset_name:
        return "tpch_sf2_z3_lineitem"
    elif "tpch_sf2_z4_lineitem" in dataset_name:
        return "tpch_sf2_z4_lineitem"
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

class Datasets(StrEnum):
    FOREST = auto()
    POWER = auto()
    CENSUS = auto()
    DMV = auto()
    TPCH = auto()
    CORRELATED = auto()
    SKEW_02 = auto()
    SKEW_04 = auto()
    SKEW_06 = auto()
    SKEW_08 = auto()
    

    def get_sql_file_path(self):
        return f"./workloads/{self.value}/{self.value}.sql"
    
    def get_database_name(self):
        if "forest" in self.value:
            return "forest"
        elif "power" in self.value:
            return "power"
        elif "census" in self.value:
            return "census"
        elif "dmv" in self.value:
            return "dmv"
        elif "tpch" in self.value:
            return "tpch"
        elif "correlated" in self.value:
            return "correlated"
        else:
            raise ValueError(f"Invalid dataset name: {self.value}")