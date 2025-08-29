# justfile for dataset download, installation, training, and testing
set working-directory := ''

default_epochs := "25"
db_name := ""

[private]
default:
  just --list

# set environment variables
env:
    #! /bin/bash
    echo "Setting environment variables"
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
train dataset_name="forest" no_of_training_queries="0": install
    #! /bin/bash
    # Conditionally build suffix
    if [ "{{no_of_training_queries}}" = "0" ]; then
        suffix=""
    else
        suffix="_notq_{{no_of_training_queries}}"
    fi
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split train --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_train_sample${suffix}.xlsx --theta_cache_path "theta_cache.pkl" \
    --cdf_cache_name "cdf_cache.pkl" --no_of_training_queries {{no_of_training_queries}}

    uv run src/residual_model_train.py --dataset_name {{dataset_name}} \
    --train_excel_path data/excels/{{dataset_name}}/dvine_v1_{{dataset_name}}_train_sample${suffix}.xlsx \
    --epochs {{default_epochs}} --output_model_name "error_comp_model${suffix}.pt"

# Test the model
test dataset_name="forest" no_of_training_queries="0": install
    #! /bin/bash
    if [ "{{no_of_training_queries}}" = "0" ]; then
        suffix=""
    else
        suffix="_notq_{{no_of_training_queries}}"
    fi
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample${suffix}.xlsx \
    --model_name "error_comp_model${suffix}.pt" \
    --theta_cache_path "theta_cache.pkl" --cdf_cache_name "cdf_cache.pkl"

train_test dataset_name="forest" no_of_training_queries="0": install
    just train {{dataset_name}} {{no_of_training_queries}}
    just test {{dataset_name}} {{no_of_training_queries}}

tt_list dataset_list="forest" no_of_training_queries_list="100 200 300": install
    #! /bin/bash
    read -ra datasets <<< "{{dataset_list}}"
    read -ra queries <<< "{{no_of_training_queries_list}}"
    for dataset in "${datasets[@]}"; do
        for no_of_training_queries in "${queries[@]}"; do
            just test "$dataset" "$no_of_training_queries"
        done
    done

# Retrain the residual model
retrain dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split train --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_train_sample_retrained_{{update-type}}.xlsx \
    --theta_cache_path "theta_cache_{{update-type}}.pkl" --cdf_cache_name "cdf_cache_{{update-type}}.pkl" --update_type {{update-type}}

    uv run src/residual_model_train.py --dataset_name {{dataset_name}} \
    --train_excel_path data/excels/dvine_v1_{{dataset_name}}_train_sample_retrained_{{update-type}}.xlsx \
    --pretrained_model_name "error_comp_model.pt" --output_model_name "error_comp_model_retrained_{{update-type}}.pt" \
    --update_type {{update-type}} --epochs 10 --step_epochs 3
    # --freeze_layer_count 2 --lr 0.0001 --tolerance 0.1


    # uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    # --output_excel_name dvine_v1_{{dataset_name}}_eval_sample_retrained_{{update-type}}.xlsx \
    # --theta_cache_path "theta_cache_{{update-type}}.pkl" --cdf_cache_name "cdf_cache_{{update-type}}.pkl" --update_type {{update-type}}


    # uv run src/residual_model_test.py --dataset_name {{dataset_name}} \
    # --train_excel_path data/excels/dvine_v1_{{dataset_name}}_train_sample_retrained_{{update-type}}.xlsx \
    # --test_excel_path data/excels/dvine_v1_{{dataset_name}}_eval_sample_retrained_{{update-type}}.xlsx \
    # --pretrained_model_name "error_comp_model_retrained_{{update-type}}.pt" --update_type {{update-type}}

# --freeze_layer_count 1 --lr 0.0001 --tolerance 0.01

# Test with the updated copula model & residual model
retest dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample_retrained_{{update-type}}.xlsx --model_name "error_comp_model_retrained_{{update-type}}.pt" \
    --theta_cache_path "theta_cache_{{update-type}}.pkl" --cdf_cache_name "cdf_cache_{{update-type}}.pkl" --update_type {{update-type}}

# Test with the updated copula model & residual model
retrain_test dataset_name="forest" update-type="ind_0.2": install
    just retrain {{dataset_name}} {{update-type}}
    just retest {{dataset_name}} {{update-type}}


# Test with the existing copula model & residual model
test-existing dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample_{{update-type}}.xlsx --model_name "error_comp_model.pt" \
    --theta_cache_path "theta_cache.pkl" --cdf_cache_name "cdf_cache.pkl" --update_type {{update-type}}

# Test with the updated copula model & existing residual model
test-updated dataset_name="forest" update-type="ind_0.2": install
    uv run src/dvine_copula_recursive_dynamic_v2.py --data_split test --dataset_name {{dataset_name}} \
    --output_excel_name dvine_v1_{{dataset_name}}_test_sample_updated_{{update-type}}.xlsx --model_name "error_comp_model.pt" \
    --theta_cache_path "theta_cache_{{update-type}}.pkl" --cdf_cache_name "cdf_cache_{{update-type}}.pkl" --update_type {{update-type}}

# Run all the commands
all: install download train test

# Run the prerequisites script

prepare-data dataset_name="forest" model_name="dvine" update_type="ind_0.2": install env
    uv run scripts/py/prerequisists.py {{dataset_name}} {{model_name}} {{update_type}}

# Build the postgres docker images
build-postgres:
    make docker

# Run the postgres docker images
run-postgres dataset_name="forest" update_type="":
    #! /bin/bash
    if [ -z "{{update_type}}" ]; then
        db_name="{{dataset_name}}"
    else
        db_name="{{dataset_name}}_{{update_type}}"
    fi

    make stop_all_containers
    make docker_run DATABASE_NAME=$db_name
    make copy_estimations DATABASE_NAME=$db_name
    make set_docker_permissions DATABASE_NAME=$db_name
    make create_db DATABASE_NAME=$db_name

# Get accuracy for all estimations
get-accuracy dataset_name="forest" update_type="":
    #! /bin/bash
    if [ -z "{{update_type}}" ]; then
        db_name="{{dataset_name}}"
    else
        db_name="{{dataset_name}}_{{update_type}}"
    fi
    make test_all_files DATABASE_NAME=$db_name

# Calculate the p-error
calculate-p-error dataset_name="forest" update_type="":
    #! /bin/bash
    if [ -z "{{update_type}}" ]; then
        db_name="{{dataset_name}}"
    else
        db_name="{{dataset_name}}_{{update_type}}"
    fi
    make p_error DATABASE_NAME=$db_name

# Run the performance script
run-postgres-performance dataset_name="forest" update_type="":
    just run-postgres {{dataset_name}} {{update_type}}
    just get-accuracy {{dataset_name}} {{update_type}}
    just calculate-p-error {{dataset_name}} {{update_type}}


# Compute per-column query usage statistics
query-stats dataset_name="forest" update-type="": install env
    #! /bin/bash
    if [ -z "{{update-type}}" ]; then
        uv run src/query_explorer.py --dataset {{dataset_name}}
    else
        uv run src/query_explorer.py --dataset {{dataset_name}} --data-updates {{update-type}}
    fi



# clear the cache
[confirm]
clear:
    rm -rf data/cdf_cache
    rm -rf data/theta_cache
    rm -rf data/data_conversion_params

# delete the data and the venv
[confirm]
delete:
    rm -rf data/
    rm -rf .venv/