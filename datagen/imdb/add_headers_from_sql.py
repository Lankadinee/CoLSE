#!/usr/bin/env python3
import argparse
import os
import re
import sys
import tempfile
from typing import Dict, List, Tuple

SCHEMA_FILENAME = "schematext.sql"
CSV_EXTENSION = ".csv"

CREATE_TABLE_REGEX = re.compile(r"^CREATE TABLE\s+([a-zA-Z0-9_]+)\s*\(")
COLUMN_LINE_REGEX = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+.+?(,)?\s*$")
END_TABLE_REGEX = re.compile(r"^\);\s*$")


def parse_schema(schema_path: str) -> Dict[str, List[str]]:
    """Parse the SQL schema file and return mapping of table name to ordered column names.

    This function extracts column names appearing between each CREATE TABLE ... ( and );
    It ignores constraints and types, taking the first token on each column line.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    table_to_columns: Dict[str, List[str]] = {}
    current_table: str = ""
    in_table: bool = False

    with open(schema_path, "r", encoding="utf-8") as schema_file:
        for raw_line in schema_file:
            line = raw_line.rstrip("\n")

            if not in_table:
                # Match CREATE TABLE line
                m = re.match(r"^CREATE TABLE\s+([a-zA-Z0-9_]+)\s*\($", line)
                if m:
                    current_table = m.group(1)
                    table_to_columns[current_table] = []
                    in_table = True
                continue

            # If we are inside a table definition, look for end or column lines
            if END_TABLE_REGEX.match(line):
                in_table = False
                current_table = ""
                continue

            # Skip empty lines and lines that start with constraints (rare in this schema)
            if not line.strip():
                continue

            # Line like: "id integer NOT NULL PRIMARY KEY," or "name character varying" etc.
            col_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s+", line)
            if col_match and current_table:
                column_name = col_match.group(1)
                # Filter out lines that are not actual columns (e.g., if a constraint started with a word)
                # In our schema, this should be fine; but guard against e.g. 'PRIMARY KEY (...)'
                if column_name.lower() not in {"primary", "foreign", "unique", "constraint"}:
                    table_to_columns[current_table].append(column_name)

    # Remove any tables that have zero columns (should not happen)
    table_to_columns = {t: cols for t, cols in table_to_columns.items() if cols}
    return table_to_columns


def detect_if_header(line: str, expected_columns: List[str], delimiter: str) -> bool:
    """Heuristically decide if the first line already equals the expected header."""
    if not line:
        return False
    first_row = [part.strip() for part in line.rstrip("\n\r").split(delimiter)]
    return first_row == expected_columns


def add_header_to_csv(csv_path: str, header: List[str], delimiter: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Prepend header to the CSV file if not present.

    Returns (changed, message)
    """
    if not header:
        return False, "No header provided"

    # Read the first line to check if header exists and to count columns
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()

    if detect_if_header(first_line, header, delimiter):
        return False, "Header already present"

    # Optionally, validate column count against first data row
    if first_line:
        data_fields = first_line.rstrip("\n\r").split(delimiter)
        if len(data_fields) != len(header):
            # We still proceed but warn the user; delimiter may differ or data malformed
            pass

    if dry_run:
        return True, "Would add header (dry-run)"

    header_line = delimiter.join(header) + "\n"

    # Write to a temp file and replace atomically
    dir_name = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, prefix=f".{base_name}.", suffix=".tmp", encoding="utf-8", newline="") as tmp:
        tmp_path = tmp.name
        tmp.write(header_line)
        with open(csv_path, "r", encoding="utf-8", newline="") as src:
            # Write the original content unchanged
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

    os.replace(tmp_path, csv_path)
    return True, "Header added"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add headers to CSV files based on SQL schema in the same directory.")
    parser.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)), help="Directory containing CSV files and schematext.sql")
    parser.add_argument("--schema", default=SCHEMA_FILENAME, help="Schema SQL filename (default: schematext.sql)")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter to use for the header (default: ',')")
    parser.add_argument("--force", action="store_true", help="Force adding header regardless of column count mismatch")
    parser.add_argument("--dry-run", action="store_true", help="Only report actions without modifying files")

    args = parser.parse_args()
    target_dir = os.path.abspath(args.dir)
    schema_path = os.path.join(target_dir, args.schema)

    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        table_to_columns = parse_schema(schema_path)
    except Exception as exc:
        print(f"Failed to parse schema: {exc}", file=sys.stderr)
        sys.exit(1)

    # Process CSV files
    processed: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    for entry in sorted(os.listdir(target_dir)):
        if not entry.endswith(CSV_EXTENSION):
            continue
        csv_path = os.path.join(target_dir, entry)
        table_name = entry[: -len(CSV_EXTENSION)]

        if table_name not in table_to_columns:
            skipped.append((entry, "No matching table in schema"))
            continue

        header = table_to_columns[table_name]

        try:
            # If not forcing, do a quick column count sanity check
            should_write = True
            if not args.force:
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    first_line = f.readline()
                if first_line:
                    num_fields = len(first_line.rstrip("\n\r").split(args.delimiter))
                    if num_fields != len(header):
                        skipped.append((entry, f"Column count mismatch: data={num_fields}, header={len(header)}; use --force to override"))
                        should_write = False
            if not should_write:
                continue

            changed, message = add_header_to_csv(csv_path, header, args.delimiter, dry_run=args.dry_run)
            if changed:
                processed.append((entry, message))
            else:
                skipped.append((entry, message))
        except Exception as exc:
            failed.append((entry, str(exc)))

    # Report
    for entry, msg in processed:
        print(f"[UPDATED] {entry}: {msg}")
    for entry, msg in skipped:
        print(f"[SKIPPED] {entry}: {msg}")
    for entry, msg in failed:
        print(f"[FAILED] {entry}: {msg}")

    # Exit code: 0 if no failures
    sys.exit(0 if not failed else 2)


if __name__ == "__main__":
    main() 