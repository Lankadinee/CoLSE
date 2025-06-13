from colse.dataset_names import DatasetNames
from colse.datasets.dataset_tpch_utils import tpch_lineitem_preprocess


def preprocess_dataset(dataset_type: DatasetNames, skip_if_exists: bool):

    if dataset_type.is_tpch_type():
        return tpch_lineitem_preprocess(dataset_type, skip_if_exists)
