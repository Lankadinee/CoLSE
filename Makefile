.PHONY: train test install download

permissions:
	chmod +x data/download.sh

download:
	./download.sh

install:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync

train:
	uv run src/dvine_copula_recursive_dynamic.py --data_split train --dataset_name forest
	uv run src/residual_model_train.py --dataset_name forest --train_excel_path data/excels/dvine_v1_forest_train_sample_auto_max_25000.xlsx

test:
	uv run src/dvine_copula_recursive_dynamic.py --data_split test --dataset_name forest 

all: install download train test