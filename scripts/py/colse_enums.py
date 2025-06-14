


from enum import StrEnum, auto


class Datasets(StrEnum):
    FOREST = auto()
    POWER = auto()
    CENSUS = auto()
    DMV = auto()
    TPCH = auto()
    CORRELATED = auto()

    def get_sql_file_path(self):
        return f"./workloads/{self.value}/{self.value}.sql"