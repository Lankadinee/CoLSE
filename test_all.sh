#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# DATASET_NAMES=()
# for dir in workloads/*/; do
#   DATASET_NAMES+=( "$(basename "$dir")" )
# done

DATASET_NAMES=(
  # "census"
  "census_cor_0.2"
  "census_ind_0.2"
  "census_skew_0.2"
  # "correlated_02"
  # "correlated_04"
  # "correlated_06"
  # "correlated_08"
  # "dmv"
  "dmv_cor_0.2"
  "dmv_ind_0.2"
  # "dmv_mixed_ratio25"
  # "dmv_mixed_ratio50"
  # "dmv_mixed_ratio75"
  "dmv_skew_0.2"
  # "forest"
  "forest_cor_0.2"
  "forest_ind_0.2"
  "forest_skew_0.2"
  # "mixed_ratio25"
  # "mixed_ratio50"
  # "mixed_ratio75"
  # "power"
  "power_cor_0.2"
  "power_ind_0.2"
  # "power_mixed_ratio25"
  # "power_mixed_ratio50"
  # "power_mixed_ratio75"
  "power_skew_0.2"
  # "tpch_lineitem_10"
  # "tpch_lineitem_20"
  # "tpch_sf2_z1_lineitem"
  # "tpch_sf2_z2_lineitem"
  # "tpch_sf2_z3_lineitem"
  # "tpch_sf2_z4_lineitem"
  # "tpch_sf2_z4_lineitem_mixed_ratio25"
  # "tpch_sf2_z4_lineitem_mixed_ratio50"
  # "tpch_sf2_z4_lineitem_mixed_ratio75"

)

for dataset_name in "${DATASET_NAMES[@]}"; do
  echo "--------------------------------"
  echo "Initializing $dataset_name"
  make init DATABASE_NAME="$dataset_name"
  echo "--------------------------------"
  echo "Testing $dataset_name"
  make test_all DATABASE_NAME="$dataset_name"
  bash clean_dockers.sh
  echo "--------------------------------"
done

for dataset_name in "${DATASET_NAMES[@]}"; do
  echo "--------------------------------"
  echo "Calculating p-error for $dataset_name"
  make p_error DATABASE_NAME="$dataset_name"
  echo "--------------------------------"
done

