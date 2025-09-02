#!/usr/bin/env python3
"""
Script to append estimates from CSV to SQL queries.
Reads forest.sql and true_card.csv, then appends the estimate to each query.
"""

import csv
import os

def append_estimates_to_sql():
    # File paths
    sql_file_path = "/datadrive500/CoLSE/workloads/forest/forest_train.sql"
    csv_file_path = "/datadrive500/CoLSE/data/forest/label_train.csv"
    output_file_path = "/datadrive500/CoLSE/workloads/forest/forest_train_with_estimates.sql"
    
    # Check if files exist
    if not os.path.exists(sql_file_path):
        print(f"Error: SQL file not found at {sql_file_path}")
        return
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    
    # Read estimates from CSV
    estimates = []
    try:
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = list(csv_reader)
            # If there is a header, start from the second row
            if rows and any(cell.lower() in ['cardinality', 'estimate'] for cell in rows[0]):
                data_rows = rows[1:]
            else:
                data_rows = rows
            for row in data_rows:
                if row:  # Check if row is not empty
                    estimates.append(row[0])  # First column contains the estimate
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Read {len(estimates)} estimates from CSV")
    
    # Read SQL file and append estimates
    try:
        with open(sql_file_path, 'r') as sqlfile:
            sql_content = sqlfile.read()
        
        # Split by semicolon to get individual queries
        queries = sql_content.split(';')
        
        # Remove empty queries (from trailing semicolon)
        queries = [q.strip() for q in queries if q.strip()]
        
        print(f"Found {len(queries)} queries in SQL file")
        
        if len(queries) != len(estimates):
            print(f"Warning: Number of queries ({len(queries)}) doesn't match number of estimates ({len(estimates)})")
            # Use the smaller of the two
            min_count = min(len(queries), len(estimates))
            queries = queries[:min_count]
            estimates = estimates[:min_count]
        
        # Create new content with estimates appended
        new_content = ""
        for i, (query, estimate) in enumerate(zip(queries, estimates)):
            # Remove trailing semicolon if present and add estimate
            query = query.rstrip(';').strip()
            new_content += f"{query} || {estimate};\n"
        
        # Write to output file
        with open(output_file_path, 'w') as outputfile:
            outputfile.write(new_content)
        
        print(f"Successfully created {output_file_path}")
        print(f"Processed {len(queries)} queries")
        
    except Exception as e:
        print(f"Error processing SQL file: {e}")
        return

if __name__ == "__main__":
    append_estimates_to_sql()
