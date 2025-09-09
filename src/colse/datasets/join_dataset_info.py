



from typing import List
from colse.dataset_names import DatasetNames


def get_all_columns(dataset_type: DatasetNames):
    if dataset_type == DatasetNames.IMDB_DATA:
        from colse.datasets.dataset_imdb import get_all_columns
    elif dataset_type == DatasetNames.CUSTOM_JOIN_DATA:
        from colse.datasets.dataset_custom_join import get_all_columns
    else:
        raise ValueError(f"Dataset {dataset_type} not supported")
    return get_all_columns()

def get_table_cols(dataset_type: DatasetNames):
    if dataset_type == DatasetNames.IMDB_DATA:
        from colse.datasets.dataset_imdb import TABLE_COLS
    elif dataset_type == DatasetNames.CUSTOM_JOIN_DATA:
        from colse.datasets.dataset_custom_join import TABLE_COLS
    else:
        raise ValueError(f"Dataset {dataset_type} not supported")
    return TABLE_COLS

def get_no_of_cols(dataset_type: DatasetNames):
    if dataset_type == DatasetNames.IMDB_DATA:
        from colse.datasets.dataset_imdb import NO_OF_COLS
    elif dataset_type == DatasetNames.CUSTOM_JOIN_DATA:
        from colse.datasets.dataset_custom_join import NO_OF_COLS
    else:
        raise ValueError(f"Dataset {dataset_type} not supported")
    return NO_OF_COLS


def get_query_indexes(dataset_type: DatasetNames, talbe_name: str, column_name: str | None = None) -> List[int]:
    index_list = []
    query_col_list = get_all_columns(dataset_type)
    for column in get_table_cols(dataset_type)[talbe_name][1:]:
        if column_name is not None and column_name != column:
            continue
        q_col_name = f"{talbe_name}:{column}"
        q_index = query_col_list.index(q_col_name)
        lb_query_index = q_index * 2
        ub_query_index = q_index * 2 + 1
        index_list.extend([lb_query_index, ub_query_index])
    return index_list