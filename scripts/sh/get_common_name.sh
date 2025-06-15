#!/bin/bash

DATABASE_NAME=$1

if [[ "$DATABASE_NAME" == *"forest"* ]]; then
    echo "forest"
elif [[ "$DATABASE_NAME" == *"census"* ]]; then
    echo "census"
elif [[ "$DATABASE_NAME" == *"power"* ]]; then
    echo "power"
elif [[ "$DATABASE_NAME" == *"dmv"* ]]; then
    echo "dmv"
elif [[ "$DATABASE_NAME" == *"tpch"* ]]; then
    echo "tpch"
elif [[ "$DATABASE_NAME" == *"correlated_02"* ]]; then
    echo "correlated_02"
elif [[ "$DATABASE_NAME" == *"correlated_04"* ]]; then
    echo "correlated_04"
elif [[ "$DATABASE_NAME" == *"correlated_06"* ]]; then
    echo "correlated_06"
elif [[ "$DATABASE_NAME" == *"correlated_08"* ]]; then
    echo "correlated_08"
elif [[ "$DATABASE_NAME" == *"tpch_lineitem_10"* ]]; then
    echo "tpch_lineitem_10"
elif [[ "$DATABASE_NAME" == *"tpch_lineitem_20"* ]]; then
    echo "tpch_lineitem_20"
elif [[ "$DATABASE_NAME" == *"tpch_sf2_z1_lineitem"* ]]; then
    echo "tpch_sf2_z1_lineitem"
elif [[ "$DATABASE_NAME" == *"tpch_sf2_z2_lineitem"* ]]; then
    echo "tpch_sf2_z2_lineitem"
elif [[ "$DATABASE_NAME" == *"tpch_sf2_z3_lineitem"* ]]; then
    echo "tpch_sf2_z3_lineitem"
elif [[ "$DATABASE_NAME" == *"tpch_sf2_z4_lineitem"* ]]; then
    echo "tpch_sf2_z4_lineitem"
else
    echo "unknown"
    exit 1
fi