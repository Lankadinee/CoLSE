


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


def _print_num_cols_distribution(split_name: str, num_cols_counter: Counter, num_queries: int) -> None:
    print(f"\n=== Split: {split_name} (num columns per query) ===")
    if num_queries == 0:
        print("(no queries)")
        return

    # Sort by number of columns ascending
    sorted_items = sorted(num_cols_counter.items(), key=lambda kv: kv[0])

    header_col = "#Columns"
    print(f"{header_col:>9}  Count  % of queries")
    print(f"{'-' * 9}  {'-' * 5}  {'-' * 12}")
    for num_cols, count in sorted_items:
        pct = (count / num_queries) * 100.0
        print(f"{str(num_cols):>9}  {str(count).rjust(5)}  {pct:10.2f}%")


def process(dataset_name: str, data_updates: Optional[str] = None, split_dict: Optional[Dict[str, int]] = None):
    query_json = load_query(dataset_name, data_updates)

    split_to_counts: Dict[str, Counter] = {}
    split_to_num_cols_dist: Dict[str, Counter] = {}

    for split, queries in query_json.items():
        no_of_queries = 0
        counts: Counter = Counter()
        num_cols_counter: Counter = Counter()
        for q in tqdm(queries, desc=f"Processing {split}"):
            if split_dict is not None and split in split_dict and no_of_queries >= split_dict[split]:
                break
            no_of_queries += 1
            # Each entry appears to be a list where the first element is a dict of column conditions
            cols_source = None
            if isinstance(q, list) and len(q) > 0 and isinstance(q[0], dict):
                cols_source = q[0]
            elif isinstance(q, dict):
                cols_source = q

            if not isinstance(cols_source, dict):
                continue
            
            no_of_query_columns = 0
            for col_name in cols_source.keys():
                if cols_source[col_name] is not None:
                    no_of_query_columns += 1
                    counts[col_name] += 1
            # Track distribution of number of columns per query
            num_cols_counter[no_of_query_columns] += 1

        split_to_counts[split] = counts
        split_to_num_cols_dist[split] = num_cols_counter

    # Pretty print stats per split
    for split, counts in split_to_counts.items():
        num_queries = len(query_json.get(split, []))
        _print_split_stats(split, counts, num_queries=num_queries)
        _print_num_cols_distribution(split, split_to_num_cols_dist[split], num_queries=num_queries)

    return split_to_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-column query usage statistics")
    parser.add_argument("--dataset", type=str, default="forest", help="Dataset name, e.g., forest")
    parser.add_argument("--data-updates", type=str, default=None, help="Optional data updates key, e.g., ind_0.2")
    parser.add_argument("--show-num-cols-dist", action="store_true", help="Also print distribution of number of columns per query")
    parser.add_argument("--no-of-queries-list", type=str, default="", help="Number of queries to process")
    parser.add_argument("--data-split-list", type=str, default="", help="Number of queries to process")
    
    args = parser.parse_args()

    no_of_queries_list = args.no_of_queries_list.strip()
    data_split_list = args.data_split_list.strip()
    if no_of_queries_list != "" and data_split_list != "":
        no_of_queries_list = [int(no_of_queries) for no_of_queries in no_of_queries_list.split(",")]
        data_split_list = data_split_list.split(",")
        assert len(no_of_queries_list) == len(data_split_list), "Number of queries and data splits must be the same"
        split_dict = dict(zip(data_split_list, no_of_queries_list))
    else:
        split_dict = None
    print(f"Split dict: {split_dict}")
    process(args.dataset, args.data_updates, split_dict)