from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from rich.console import Console
from rich.table import Table


class Args:
    def __init__(self, **kwargs):
        self.dataset_name = kwargs.get("dataset_name", "forest")
        self.bs = 32
        self.epochs = 25
        self.lr = 0.001  # default value in both pytorch and keras
        self.hid_units = "256_256_128_64"
        self.no_of_queries = -1
        self.additional_features = 2
        self.dropout_prob = None
        self.output_len = 3
        self.train_test_split = 0.8
        self.step_epochs = 10
        self.freeze_layer_count = 0

        self.train_excel_path = (
            get_data_path() / "excels/dvine_v1_forest_train_sample_auto_max_25000.xlsx"
        )
        self.test_excel_path = None
        self.valid_excel_path = None

        # overwrite parameters from user
        self.__dict__.update(kwargs)

        self.dataset = DatasetNames(self.dataset_name)
        if self.dataset.is_join_type():
            self.fea_num = 8 * 2 + 6 + 1
        else:
            self.fea_num = self.dataset.get_no_of_columns() * 2 + self.additional_features
        self.update_type = kwargs.get("update_type", None)

    def __str__(self):
        return f"Args: {self.__dict__}"

    def get_hyperparameters(self):
        return {
            "dataset_name": self.dataset_name,
            "bs": self.bs,
            "epochs": self.epochs,
            "lr": self.lr,
            "hid_units": self.hid_units,
            "no_of_queries": self.no_of_queries,
            "additional_features": self.additional_features,
            "dropout_prob": self.dropout_prob,
            "output_len": self.output_len,
            "train_test_split": self.train_test_split,
            "step_epochs": self.step_epochs,
            "freeze_layer_count": self.freeze_layer_count,
            "tolerance": self.tolerance,
        }
    
    def show_args(self):
        table = Table(title="Args")
        table.add_column("Name", style="cyan")
        table.add_column("Value", style="magenta")
        for key, value in self.get_hyperparameters().items():
            table.add_row(key, str(value))
        console = Console()
        console.print(table)
