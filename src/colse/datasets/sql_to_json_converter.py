#!/usr/bin/env python3
"""
Convert IMDB SQL queries to JSON format similar to forest dataset


Example query:
SELECT COUNT(*) FROM customers c,orders o WHERE c.customer_id=o.customer_id AND o.amount>300;

Note that
1. no space between 'customers c,orders o' c and orders
2. =< and >= are not supported
"""

import argparse
import json
import re

from colse.datasets.dataset_custom_join import (
    get_all_columns as get_all_columns_custom_join,
)
from colse.datasets.dataset_imdb import get_all_columns as get_all_columns_imdb


def parse_sql_query(expected_columns, sql_query):
    """Parse a SQL query and extract constraints"""
    constraints = {}

    # Initialize all columns with null
    for column in expected_columns:
        constraints[column] = None

    # Extract table aliases and their mappings
    table_mappings = {}
    from_match = re.search(r"FROM\s+(.+?)\s+WHERE", sql_query, re.IGNORECASE)
    if from_match:
        tables = from_match.group(1).split(",")
        for table in tables:
            parts = table.strip().split()
            if len(parts) >= 2:
                alias = parts[1].strip()
                table_name = parts[0].strip()
                table_mappings[alias] = table_name

    # Extract WHERE conditions
    where_match = re.search(r"WHERE\s+(.+?)(?:;|$)", sql_query, re.IGNORECASE)
    if where_match:
        conditions = where_match.group(1).split("AND")

        for condition in conditions:
            condition = condition.strip()

            # Handle different types of conditions
            if "=" in condition:
                # Equality condition
                parts = condition.split("=")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    # Extract column name and value
                    if "." in left:
                        table_alias, column = left.split(".")
                        if table_alias in table_mappings:
                            table_name = table_mappings[table_alias]
                            full_column = f"{table_name}:{column}"

                            # Try to extract numeric value
                            try:
                                value = int(right)
                                constraints[full_column] = ["=", value]
                            except ValueError:
                                # If not numeric, skip (likely a join condition)
                                pass

            elif ">" in condition:
                # Greater than condition
                parts = condition.split(">")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    if "." in left:
                        table_alias, column = left.split(".")
                        if table_alias in table_mappings:
                            table_name = table_mappings[table_alias]
                            full_column = f"{table_name}:{column}"

                            try:
                                value = int(right)
                                constraints[full_column] = [">", value]
                            except ValueError:
                                pass

            elif "<" in condition:
                # Less than condition
                parts = condition.split("<")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    if "." in left:
                        table_alias, column = left.split(".")
                        if table_alias in table_mappings:
                            table_name = table_mappings[table_alias]
                            full_column = f"{table_name}:{column}"

                            try:
                                value = int(right)
                                constraints[full_column] = ["<", value]
                            except ValueError:
                                pass

    # Create ordered constraints dictionary
    ordered_constraints = {}
    for column in expected_columns:
        ordered_constraints[column] = constraints[column]

    return ordered_constraints


def parse_joined_tables(sql_query):
    """Parse the joined tables in a SQL query"""
    from_match = re.search(r"FROM\s+(.+?)\s+WHERE", sql_query, re.IGNORECASE)
    if from_match:
        tables = [t.split(" ")[0] for t in from_match.group(1).split(",")]
        return tables
    return []


def convert_sql_to_json(database_name, sql_file_path, output_file_path):
    """Convert SQL file to JSON format"""
    # Define the expected columns in order
    if database_name == "imdb":
        expected_columns = get_all_columns_imdb()
    elif database_name == "custom_join":
        expected_columns = get_all_columns_custom_join()
    else:
        raise ValueError(f"Database {database_name} not supported")

    # Read SQL file
    with open(sql_file_path, "r") as f:
        sql_content = f.read()

    # Split into individual queries
    queries = [q.strip() for q in sql_content.split(";") if q.strip()]

    # Convert each query
    json_data = {"train": []}

    for i, query in enumerate(queries):
        if not query:
            continue

        constraints = parse_sql_query(expected_columns, query)
        joined_tables = parse_joined_tables(query)

        # Create JSON entry
        entry = [constraints, joined_tables, i + 1]  # Using index+1 as target value
        json_data["train"].append(entry)

    # Write JSON file
    with open(output_file_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Converted {len(json_data['train'])} queries to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SQL queries to JSON format for supported datasets."
    )
    parser.add_argument(
        "--database",
        "-d",
        choices=["imdb", "custom_join"],
        default="custom_join",
        help="Name of the database schema to use. Defaults to 'imdb'.",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_file",
        help="Path to input .sql file. Defaults to data/<database>/job-light.sql",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        help="Path to output .json file. Defaults to data/<database>/job-light.json",
    )

    args = parser.parse_args()

    database_name = args.database
    default_input = f"data/{database_name}/job-light.sql"
    default_output = f"data/{database_name}/job-light.json"
    input_file = args.input_file or default_input
    output_file = args.output_file or default_output

    convert_sql_to_json(database_name, input_file, output_file)
