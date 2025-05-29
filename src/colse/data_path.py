from pathlib import Path


def get_data_path(post_fix=None):
    CWD = Path(__file__).resolve().parent
    data_path = CWD / "../../data"
    if post_fix is not None:
        data_path = data_path / post_fix
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