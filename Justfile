# justfile for dataset download, installation, training, and testing
set working-directory := ''

default_epochs := "25"

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
train dataset_name="forest": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split train --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_train_sample.xlsx --theta_cache_path "theta_cache.pkl" \
    --cdf_cache_name "cdf_cache.pkl"

    uv run src/residual_model_train.py --dataset_name {{dataset_name}} \
    --train_excel_path data/excels/dvine_v1_{{dataset_name}}_train_sample.xlsx --epochs {{default_epochs}} --output_model_name "error_comp_model.pt"

# Test the model
test dataset_name="forest": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample.xlsx --model_name "error_comp_model.pt" \
    --theta_cache_path "theta_cache.pkl" --cdf_cache_name "cdf_cache.pkl"

# Retrain the model
retrain dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split train --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_train_sample_retrained_{{update-type}}.xlsx \
    --theta_cache_path "theta_cache_retrained.pkl" --cdf_cache_name "cdf_cache_retrained.pkl" --update_type {{update-type}}

    uv run src/residual_model_train.py --dataset_name {{dataset_name}} \
    --train_excel_path data/excels/dvine_v1_{{dataset_name}}_train_sample_retrained_{{update-type}}.xlsx --epochs 5 \
    --pretrained_model_name "error_comp_model.pt" --output_model_name "error_comp_model_retrained.pt" 

# Test the retrained model
retest dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample_retrained_{{update-type}}.xlsx --model_name "error_comp_model_retrained.pt" \
    --theta_cache_path "theta_cache_retrained.pkl" --cdf_cache_name "cdf_cache_retrained.pkl" --update_type {{update-type}}

# Run all the commands
all: install download train test

clean:
    pyclean .
