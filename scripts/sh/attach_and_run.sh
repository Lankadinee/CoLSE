#!/bin/bash

# set -euxo pipefail

# Check if a container name/ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <container_name_or_id>"
    exit 1
fi

DATABASE_NAME=$1 # e.g. correlated_02, census_cor_0.2
CONTAINER_NAME=$2
DATABASE_COMMON_NAME=$3 # e.g. correlated_02, census

echo "DATABASE_NAME: $DATABASE_NAME"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "DATABASE_COMMON_NAME: $DATABASE_COMMON_NAME"

# To split DATABASE_COMMON_NAME by underscore in bash, you can use:
IFS='_' read -ra PARTS <<< "$DATABASE_COMMON_NAME"
DATABASE_SPLIT_NAME=${PARTS[0]} # e.g. correlated, census
# For example, if DATABASE_COMMON_NAME="correlated_02", then PARTS[0]="correlated" and PARTS[1]="02"

# Export the password for non-interactive login
export PGPASSWORD=postgres

# Copy estimations to the container
# echo "Copying estimations"
# for file in workloads/$DATABASE_NAME/estimates/*.csv; 
# do 
#     echo "Copying $file"; 
#     docker cp "$file" $CONTAINER_NAME:/var/lib/pgsql/13.1/data/; 
# done

# Copy the single table datasets to the container
# docker exec $CONTAINER_NAME mkdir -p /tmp/single_table_datasets; 
# docker cp $(pwd)/single_table_datasets/$DATABASE_COMMON_NAME/ $CONTAINER_NAME:/tmp/single_table_datasets/$DATABASE_COMMON_NAME; 
# docker exec $CONTAINER_NAME ls -la /tmp/single_table_datasets/$DATABASE_COMMON_NAME # show the contents of the container

# Copy the scripts to the container
# docker cp $(pwd)/scripts $CONTAINER_NAME:/tmp/scripts
# docker cp $(pwd)/scripts/sql/ $CONTAINER_NAME:/tmp/scripts/sql/

# Set the permissions for the container
# docker exec --user root $CONTAINER_NAME chown -R postgres:postgres /var/lib/pgsql/13.1/data/
# docker exec --user root $CONTAINER_NAME chmod -R 750 /var/lib/pgsql/13.1/data/
# docker exec --user root $CONTAINER_NAME chown -R postgres:postgres /tmp/scripts
# docker exec --user root $CONTAINER_NAME chown -R postgres:postgres /tmp/single_table_datasets/
# docker exec --user root $CONTAINER_NAME chmod -R 750 /tmp/scripts
# docker exec --user root $CONTAINER_NAME chmod -R 750 /tmp/single_table_datasets/

# Create a temporary SQL script file with all commands
SQL_SCRIPT="/tmp/commands.sql"
cat <<EOF >$SQL_SCRIPT
CREATE DATABASE $DATABASE_SPLIT_NAME;
\c $DATABASE_SPLIT_NAME;
\i /tmp/single_table_datasets/$DATABASE_COMMON_NAME/$DATABASE_COMMON_NAME.sql;

\i /tmp/scripts/sql/${DATABASE_SPLIT_NAME}_load.sql;
\i /tmp/scripts/sql/${DATABASE_SPLIT_NAME}_index.sql;
EOF

LOAD_SQL_SCRIPT=/tmp/${DATABASE_COMMON_NAME}_load.sql
cat <<EOF >$LOAD_SQL_SCRIPT
\copy $DATABASE_SPLIT_NAME from '/tmp/single_table_datasets/$DATABASE_COMMON_NAME/$DATABASE_NAME.csv' with CSV header;
EOF

# Create the directory in the container
docker exec $CONTAINER_NAME mkdir -p /tmp/single_table_datasets

# Copy SQL commands to the container
docker cp $SQL_SCRIPT $CONTAINER_NAME:/tmp/commands.sql
echo "Copied $SQL_SCRIPT to $CONTAINER_NAME:/tmp/commands.sql"

# Copy the load SQL script to the container
docker cp $LOAD_SQL_SCRIPT $CONTAINER_NAME:/tmp/scripts/sql/${DATABASE_SPLIT_NAME}_load.sql
echo "Copied $LOAD_SQL_SCRIPT to $CONTAINER_NAME:/tmp/scripts/sql/${DATABASE_SPLIT_NAME}_load.sql"

# Commands to execute inside the container
COMMANDS="psql -d template1 -h localhost -U postgres -f /tmp/commands.sql"

# Attach to the container and run commands
docker exec -it --user postgres  $CONTAINER_NAME bash -c "$COMMANDS"

# Cleanup: Remove the temporary SQL script
rm -f $SQL_SCRIPT
