


from colse.custom_data_generator import CustomDataGen
from colse.dataset_names import DatasetNames


class ForestDataGen(CustomDataGen):
    def __init__(self):
        super().__init__(
            no_of_rows=None,
            no_of_queries= None,
            dataset_type= DatasetNames.FOREST_DATA,
            data_split="test",
            selected_cols= None,
            scalar_type = "min_max",
            seed = 0,
            is_range_queries = True,
            VERBOSE = True,
        )