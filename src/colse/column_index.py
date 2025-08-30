


from enum import Enum, auto
from loguru import logger
import numpy as np
from dataclasses import dataclass
from pandas import DataFrame

from colse.column_ordering import mutinfo_ordering


class ColumnIndexTypes(Enum):
    DEFAULT = auto()
    NATURAL_SKIP_ORDERING = auto()
    MUTINFO_SKIP_ORDERING = auto()
    PMUTINFO_SKIP_ORDERING = auto()
    
    def is_mutinfo_ordering(self):
        return self in [ColumnIndexTypes.MUTINFO_SKIP_ORDERING, ColumnIndexTypes.PMUTINFO_SKIP_ORDERING]


class ColumnIndexProvider:
    def __init__(self, data: DataFrame, column_index_type: ColumnIndexTypes):
        self.column_index_type: ColumnIndexTypes = column_index_type
        self.col_order_mapping = []

        match column_index_type:
            case ColumnIndexTypes.MUTINFO_SKIP_ORDERING:
                column_ordering = mutinfo_ordering(data, method="MutInfo")
            case ColumnIndexTypes.PMUTINFO_SKIP_ORDERING:
                column_ordering = mutinfo_ordering(data, method="PMutInfo")
            case _:
                pass
        
        if column_index_type.is_mutinfo_ordering():
            column_names = data.columns.tolist()
            self.col_order_mapping = [column_names.index(c) for c in column_ordering]
        
        logger.info(f"Column Ordering Mapping: {self.col_order_mapping}")


    def get_skip_ordering(self, o_cdf_list):
        r_cdf_list = o_cdf_list.reshape(-1, 2)
        mask = (r_cdf_list[:, 0] != 0) | (r_cdf_list[:, 1] != 1)
        non_zero_non_one_indices = np.flatnonzero(mask)
        if non_zero_non_one_indices.size == 0:
            col_indices = []
            cdf_list = np.array([], dtype=o_cdf_list.dtype)
        else:
            col_indices = non_zero_non_one_indices + 1
            cdf_list = r_cdf_list[mask].reshape(-1)
        return list(col_indices), list(cdf_list)

    def get_column_index(self, o_cdf_list):
        no_of_columns = len(o_cdf_list) // 2
        # switch case on the column index type
        match self.column_index_type:
            case ColumnIndexTypes.DEFAULT:
                col_indices = [i + 1 for i in range(no_of_columns)]
                cdf_list = o_cdf_list.reshape(-1)
            case ColumnIndexTypes.NATURAL_SKIP_ORDERING:
                col_indices, cdf_list = self.get_skip_ordering(o_cdf_list)
            case ColumnIndexTypes.MUTINFO_SKIP_ORDERING | ColumnIndexTypes.PMUTINFO_SKIP_ORDERING:
                col_indices, cdf_list = self.get_skip_ordering(o_cdf_list)
                col_indices = [self.col_order_mapping[i-1]+1 for i in col_indices]
            
        return col_indices, cdf_list