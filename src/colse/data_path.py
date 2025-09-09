from pathlib import Path
from enum import StrEnum, auto


class DataPathDir(StrEnum):
    MODELS = auto()
    LOGS = auto()
    EXCELS = auto()
    CDF_CACHE = auto()
    THETA_CACHE = auto()
    DATAGEN_CACHE = auto()
    DATA_UPDATES = auto()
    DATA_CONVERSION_PARAMS = auto()
    WORKLOAD_UPDATES = auto()
    NPY_FILES = auto()

def get_data_path(*args):
    CWD = Path(__file__).resolve().parent
    data_path = CWD / "../../data"
    for arg in args:
        data_path = data_path / arg if arg else data_path
    if not data_path.exists():
        # create the data path
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def get_model_path(dataset_path=None):
    CWD = Path(__file__).resolve().parent
    model_path = get_data_path() / "models"
    if dataset_path is not None:
        model_path = model_path / dataset_path
    if not model_path.exists():
        # create the model path
        model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def get_log_path():
    CWD = Path(__file__).resolve().parent
    log_path = get_data_path() / "logs"
    if not log_path.exists():
        # create the log path
        log_path.mkdir(parents=True, exist_ok=True)
    return log_path

def get_excel_path():
    CWD = Path(__file__).resolve().parent
    excel_path = get_data_path() / "excels"
    if not excel_path.exists():
        # create the excel path
        excel_path.mkdir(parents=True, exist_ok=True)
    return excel_path


if __name__ == "__main__":
    print(get_data_path("test123"))
