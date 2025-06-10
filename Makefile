.PHONY: execute_permission apply_patch p_error docker stop_all_containers docker_run clean set_docker_permissions \
replace_file replace all copy_estimations create_db start_container test_one_file test_experiment test_census analyze_one_file

IMAGE_NAME=ceb
DATABASE_NAME=forest
CONTAINER_NAME=ce-benchmark-$(IMAGE_NAME)-$(DATABASE_NAME)
TEST_FILENAME=forest_true_card.csv
base_name := $(basename $(TEST_FILENAME))
ESTIMATES_COST_FILE_PATH := "scripts/plan_cost/$(DATABASE_NAME)/$(base_name)_cost.csv"
TRUE_COST_FILE_PATH="scripts/plan_cost/$(DATABASE_NAME)/$(DATABASE_NAME)_true_card_cost.csv"

execute_permission:
	@chmod u+x *.sh
	@chmod u+x scripts/sh/*.sh

create_venv:
	@uv sync

start_container:
	@docker start $(CONTAINER_NAME)
	@sleep 2
	echo "$(CONTAINER_NAME) Container started!"

stop_container:
	@docker stop $(CONTAINER_NAME)
	echo "$(CONTAINER_NAME) Container stopped!"

delete_container:
	@docker rm -f $(CONTAINER_NAME) || true
	echo "$(CONTAINER_NAME) Container deleted!"

stop_all_containers:
	echo "Stopping all containers"
	docker stop $(shell docker ps -a -q) || true

docker:
	@tar cvf postgres-13.1.tar.gz postgresql-13.1
	@mv postgres-13.1.tar.gz dockerfile/
	@cd dockerfile && docker build -t $(IMAGE_NAME) --network=host .
	@rm -rf postgres-13.1.tar.gz

docker_run:
	echo "Starting docker"
	docker rm -f $(CONTAINER_NAME) || true
	docker run --name $(CONTAINER_NAME) -p 5431:5432 -d $(IMAGE_NAME)

	# docker run -v $(shell pwd)/single_table_datasets/${DATABASE_NAME}:/tmp/single_table_datasets/${DATABASE_NAME}:ro -v $(shell pwd)/scripts:/tmp/scripts:ro --name $(CONTAINER_NAME) -p 5431:5432 -d $(IMAGE_NAME)
	echo "Docker is running"

copy_estimations:
	echo "Copying estimations"
	@for file in workloads/$(DATABASE_NAME)/estimates/*.csv; do \
		echo "Copying $$file"; \
		docker cp "$$file" $(CONTAINER_NAME):/var/lib/pgsql/13.1/data/; \
	done

	@COMMON_NAME=$$(bash scripts/sh/get_common_name.sh $(DATABASE_NAME)); \
	docker exec $(CONTAINER_NAME) mkdir -p /tmp/single_table_datasets; \
	docker cp $$(pwd)/single_table_datasets/$$COMMON_NAME/ $(CONTAINER_NAME):/tmp/single_table_datasets/$(COMMON_NAME); \
	docker cp $$(pwd)/scripts $(CONTAINER_NAME):/tmp/scripts

set_docker_permissions:
	@docker exec --user root $(CONTAINER_NAME) chown -R postgres:postgres /var/lib/pgsql/13.1/data/
	@docker exec --user root $(CONTAINER_NAME) chmod -R 750 /var/lib/pgsql/13.1/data/
	@docker exec --user root $(CONTAINER_NAME) chown -R postgres:postgres /tmp/scripts
	@docker exec --user root $(CONTAINER_NAME) chown -R postgres:postgres /tmp/single_table_datasets/
	@docker exec --user root $(CONTAINER_NAME) chmod -R 750 /tmp/scripts
	@docker exec --user root $(CONTAINER_NAME) chmod -R 750 /tmp/single_table_datasets/

create_db:
	#!/bin/bash
	@sleep 2
	@COMMON_NAME=$$(bash scripts/sh/get_common_name.sh $(DATABASE_NAME)); \
	./scripts/sh/attach_and_run.sh $(DATABASE_NAME) $(CONTAINER_NAME) $$COMMON_NAME
	# Attach to the container and run commands
	# docker exec --user postgres  $(CONTAINER_NAME) bash -c "psql -d template1 -h localhost -U postgres -f /tmp/commands.sql"

init: execute_permission stop_all_containers docker_run copy_estimations set_docker_permissions create_db  

test_one_file:
	uv run scripts/py/send_query.py --database_name $(DATABASE_NAME) --container_name $(CONTAINER_NAME) --filename $(TEST_FILENAME) 2>&1 | tee -a $(DATABASE_NAME)_test.log

test_one: stop_all_containers start_container create_venv test_one_file 

p_error:
	uv run scripts/py/p_error_calculation.py --database_name $(DATABASE_NAME)
	mkdir -p scripts/plan_cost/$(DATABASE_NAME)/results
	mv scripts/plan_cost/$(DATABASE_NAME)/*.txt scripts/plan_cost/$(DATABASE_NAME)/results/

show_logs:
	@docker exec $(CONTAINER_NAME) cat /var/lib/pgsql/13.1/data/custom_log_file.txt

reset_logs:
	@docker exec $(CONTAINER_NAME) rm /var/lib/pgsql/13.1/data/custom_log_file.txt || true


