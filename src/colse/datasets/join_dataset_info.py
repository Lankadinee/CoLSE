



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