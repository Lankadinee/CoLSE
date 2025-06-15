


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
    elif "correlated" in dataset_name:
        return "correlated"
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