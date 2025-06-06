# justfile for dataset download, installation, training, and testing
set working-directory := ''

default:
  just --list

# set environment variables
env:
    #! /bin/bash
    export PYTHONPATH=$(pwd)/src

performance: env
    #! /bin/bash
    export PYTHONPATH=$(pwd)/src
    uv run scripts/dequantizer_perfoemance.py

# Set permissions for the download script
permissions:
    chmod +x download.sh

# Download the dataset
download: permissions
    ./download.sh

# Install the dependencies
install:
    which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync

# Train the model
train dataset_name="forest" epochs="25":
    uv run src/dvine_copula_recursive_dynamic.py --data_split train --dataset_name {{dataset_name}}
    # uv run src/residual_model_train.py --dataset_name {{dataset_name}} --train_excel_path data/excels/dvine_v1_{{dataset_name}}_train_sample.xlsx --epochs {{epochs}}

# Test the model
test dataset_name="forest":
    uv run src/dvine_copula_recursive_dynamic.py --data_split test --dataset_name {{dataset_name}}

# Run all the commands
all: install download train test
