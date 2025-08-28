


from collections import defaultdict, Counter
import json
import argparse
from typing import Optional, Dict

from tqdm import tqdm
from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames


def load_query(dataset_name: str, data_updates: Optional[str] = None):
    if data_updates is None:
        file_path = get_data_path(f"{dataset_name}/query.json")
    else:
        if "ratio" in data_updates:
            file_path = get_data_path(f"{dataset_name}/workload_updates/query_mixed_{data_updates}.json")
        else:
            file_path = get_data_path(f"{dataset_name}/data_updates/query_{data_updates}.json")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Query file not found: {file_path}")
    
    print("-"*100)
    print(f"Loading query from {file_path}")
    print("-"*100)
    with open(file_path, "r") as f:
        query = json.load(f)
    return query


def _print_split_stats(split_name: str, col_counts: Counter, num_queries: int) -> None:
    print(f"\n=== Split: {split_name} ===")
    print(f"Total queries: {num_queries}")
    print(f"Unique columns: {len(col_counts)}")

    if num_queries == 0:
        print("(no queries)")
        return

    # Sort by count desc, then by column name
    sorted_items = sorted(col_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    # Column for alignment
    max_col_len = max((len(col) for col, _ in sorted_items), default=5)
    header_col = "Column".ljust(max_col_len)
    print(f"{header_col}  Count  % of queries")
    print(f"{'-' * max_col_len}  {'-' * 5}  {'-' * 12}")

    for col, count in sorted_items:
        pct = (count / num_queries) * 100.0
        print(f"{col.ljust(max_col_len)}  {str(count).rjust(5)}  {pct:10.2f}%")


def process(dataset_name: str, data_updates: Optional[str] = None):
    query_json = load_query(dataset_name, data_updates)

    split_to_counts: Dict[str, Counter] = {}

    for split, queries in query_json.items():
        counts: Counter = Counter()
        for q in tqdm(queries, desc=f"Processing {split}"):
            # Each entry appears to be a list where the first element is a dict of column conditions
            cols_source = None
            if isinstance(q, list) and len(q) > 0 and isinstance(q[0], dict):
                cols_source = q[0]
            elif isinstance(q, dict):
                cols_source = q

            if not isinstance(cols_source, dict):
                continue

            for col_name in cols_source.keys():
                counts[col_name] += 0 if cols_source[col_name] is None else 1

        split_to_counts[split] = counts

    # Pretty print stats per split
    for split, counts in split_to_counts.items():
        _print_split_stats(split, counts, num_queries=len(query_json.get(split, [])))

    return split_to_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-column query usage statistics")
    parser.add_argument("--dataset", type=str, default="forest", help="Dataset name, e.g., forest")
    parser.add_argument("--data-updates", type=str, default=None, help="Optional data updates key, e.g., ind_0.2")
    args = parser.parse_args()

    process(args.dataset, args.data_updates)