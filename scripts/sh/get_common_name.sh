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
elif [[ "$DATABASE_NAME" == *"correlated"* ]]; then
    echo "correlated"
else
    echo "unknown"
    exit 1
fi