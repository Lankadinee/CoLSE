from colse.dataset_names import DatasetNames
from colse.datasets.dataset_tpch_utils import (
    generate_dataset_tpch_lineitem,
    get_queries_tpch_lineitem,
)


def generate_dataset_tpch_sf2_z0_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z0_LINEITEM
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_sf2_z0_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z0_LINEITEM
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_sf2_z1_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z1_LINEITEM
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_sf2_z1_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z1_LINEITEM
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_sf2_z2_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z2_LINEITEM
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_sf2_z2_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z2_LINEITEM
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_sf2_z3_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z3_LINEITEM
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_sf2_z3_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z3_LINEITEM
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_sf2_z4_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z4_LINEITEM
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_sf2_z4_lineitem(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_SF2_Z4_LINEITEM
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_lineitem_10(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_LINEITEM_10
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_lineitem_10(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_LINEITEM_10
    return get_queries_tpch_lineitem(**kwargs)


def generate_dataset_tpch_lineitem_20(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_LINEITEM_20
    return generate_dataset_tpch_lineitem(**kwargs)


def get_queries_tpch_lineitem_20(**kwargs):
    kwargs["dataset_type"] = DatasetNames.TPCH_LINEITEM_20
    return get_queries_tpch_lineitem(**kwargs)
