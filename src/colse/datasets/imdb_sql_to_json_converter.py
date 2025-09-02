#!/usr/bin/env python3
"""
Convert IMDB SQL queries to JSON format similar to forest dataset
"""

import json
import re


def parse_sql_query(sql_query):
    """Parse a SQL query and extract constraints"""
    constraints = {}

    # Define the expected columns in order
    expected_columns = [
        "cast_info:role_id",
        "movie_companies:company_id",
        "movie_companies:company_type_id",
        "movie_info:info_type_id",
        "movie_keyword:keyword_id",
        "title:kind_id",
        "title:production_year",
        "movie_info_idx:info_type_id",
    ]

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
        tables = [ t.split(' ')[0] for t in from_match.group(1).split(",") ]
        return tables
    return []

def convert_sql_to_json(sql_file_path, output_file_path):
    """Convert SQL file to JSON format"""

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

        constraints = parse_sql_query(query)
        joined_tables = parse_joined_tables(query)

        # Create JSON entry
        entry = [constraints, joined_tables, i + 1]  # Using index+1 as target value
        json_data["train"].append(entry)

    # Write JSON file
    with open(output_file_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Converted {len(json_data['train'])} queries to {output_file_path}")


if __name__ == "__main__":
    input_file = "data/imdb/job-light.sql"
    output_file = "data/imdb/job-light.json"

    convert_sql_to_json(input_file, output_file)
