#!/bin/bash

# set -euxo pipefail

# Check if a container name/ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <container_name_or_id>"
    exit 1
fi

DATABASE_NAME=$1
CONTAINER_ID=$2
DATABASE_COMMON_NAME=$3

# Export the password for non-interactive login
export PGPASSWORD=postgres

# Create a temporary SQL script file with all commands
SQL_SCRIPT="/tmp/commands.sql"
cat <<EOF >$SQL_SCRIPT
CREATE DATABASE $DATABASE_COMMON_NAME;
\c $DATABASE_COMMON_NAME;
\i /tmp/single_table_datasets/$DATABASE_COMMON_NAME/$DATABASE_COMMON_NAME.sql;
\i /tmp/scripts/sql/${DATABASE_COMMON_NAME}_load.sql;
\i /tmp/scripts/sql/${DATABASE_COMMON_NAME}_index.sql;
EOF

LOAD_SQL_SCRIPT=/tmp/${DATABASE_COMMON_NAME}_load.sql
cat <<EOF >$LOAD_SQL_SCRIPT
\copy $DATABASE_COMMON_NAME from '/tmp/single_table_datasets/$DATABASE_COMMON_NAME/$DATABASE_NAME.csv' with CSV header;
EOF

# Copy the script to the container
docker exec $CONTAINER_ID mkdir -p /tmp/single_table_datasets
docker cp $SQL_SCRIPT $CONTAINER_ID:/tmp/commands.sql
echo "Copied $SQL_SCRIPT to $CONTAINER_ID:/tmp/commands.sql"
docker cp $LOAD_SQL_SCRIPT $CONTAINER_ID:/tmp/scripts/sql/${DATABASE_COMMON_NAME}_load.sql
echo "Copied $LOAD_SQL_SCRIPT to $CONTAINER_ID:/tmp/scripts/sql/${DATABASE_COMMON_NAME}_load.sql"

# Commands to execute inside the container
COMMANDS="psql -d template1 -h localhost -U postgres -f /tmp/commands.sql"

# Attach to the container and run commands
docker exec -it --user postgres  $CONTAINER_ID bash -c "$COMMANDS"

# Cleanup: Remove the temporary SQL script
rm -f $SQL_SCRIPT
