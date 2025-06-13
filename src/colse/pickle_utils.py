import pickle
from pathlib import Path
import time


def pickle_save(object_to_save, file_name: Path):
    start_time = time.perf_counter()
    with open(f'{file_name}', 'wb') as f:
        pickle.dump(object_to_save, f)
    
    end_time = time.perf_counter()
    print(f"Pickle saved in {file_name} in {(end_time - start_time)*1000:.2f} ms")


def pickle_load(file_name: Path):
    start_time = time.perf_counter()
    if not file_name.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    # Load the pickled array
    with open(f'{file_name}', 'rb') as file:
        _loaded_obj = pickle.load(file)
    end_time = time.perf_counter()
    print(f"Pickle loaded from {file_name} in {(end_time - start_time)*1000:.2f} ms")
    return _loaded_obj