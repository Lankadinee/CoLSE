from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames


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

        self.train_excel_path = (
            get_data_path() / "excels/dvine_v1_forest_train_sample_auto_max_25000.xlsx"
        )
        self.test_excel_path = (
            get_data_path() / "excels/dvine_v1_forest_test_sample.xlsx"
        )

        # overwrite parameters from user
        self.__dict__.update(kwargs)

        self.dataset = DatasetNames(self.dataset_name)
        self.fea_num = self.dataset.get_no_of_columns() * 2 + self.additional_features

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
        }
