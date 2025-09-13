"""
Prerequisites script for dataset preparation.

Usage:
    prerequisists.py <dataset_name> <model_name> <update_type>
    prerequisists.py --help

Arguments:
    dataset_name    Name of the dataset to process
    model_name      Name of the model to use
    update_type     Type of update to apply

Options:
    -h, --help      Show this help message and exit

Examples:
    prerequisists.py imdb dvine insert
    prerequisists.py tpch ace delete
"""

import sys
from docopt import docopt
from remove_header import remove_header_if_exists

from enum import Enum
from pathlib import Path
import shutil
from loguru import logger
import pandas as pd
from colse.data_path import get_data_path
from colse.dataset_names import DatasetNames
from colse.df_utils import load_dataframe, save_dataframe
from colse.model_names import ModelNames
from colse.update_type import CustomUpdateTypes, UpdateTypes, WorkloadTypes
from py_utils import csv_to_estimates_csv, excel_to_estimates_csv, json_file_to_sql_file
from glob import glob
from rich.console import Console
from rich.table import Table


class TypeOfRetrain(str, Enum):
    NONE = ""
    RETRAINED = "retrained"
    UPDATED = "updated"

    def __str__(self):
        return self.value


CURRENT_DIR = Path(__file__).parent.parent.parent

logger.info(CURRENT_DIR)

def get_file_size_in_mb(file_path: Path) -> str:
    """Get the file size in MB."""
    if file_path.exists():
        return f"{file_path.stat().st_size / (1024 * 1024):.3f}"  # Convert bytes to MB
    return "N/A"


class Prerequisists:
    """
    This class is used to create the prerequisists for the dataset.
    It will create the queries, predictions, original dataset, and true cardinality.
    It will also create the directories for the dataset.
    """

    EXTERNAL_PRED_PATH = Path("/datadrive500/AreCELearnedYetForked")
    def __init__(self, dataset_name: str, model_name: str, up_wl_type: str):

        self.dataset_name: DatasetNames = DatasetNames(dataset_name)
        self.model_name: ModelNames = ModelNames(model_name)

        self.update_type: UpdateTypes = UpdateTypes(up_wl_type) if isinstance(up_wl_type, str) and up_wl_type in UpdateTypes else None
        self.workload_type: WorkloadTypes = WorkloadTypes(up_wl_type) if isinstance(up_wl_type, str) and up_wl_type in WorkloadTypes else None
        self.custom_update_type: CustomUpdateTypes = CustomUpdateTypes(up_wl_type) if isinstance(up_wl_type, str) and up_wl_type in CustomUpdateTypes else None

        logger.info(f"Custom update type: {self.custom_update_type} model name: {self.model_name.is_ours()}")
        assert not(self.custom_update_type and not self.model_name.is_ours()), f"Custom update type[{self.custom_update_type}] is only supported for our model"

        self.list_of_retrained_models = []

        # query paths
        if self.update_type:
            self.query_json_file_source = CURRENT_DIR / f"data/{self.dataset_name}/data_updates/query_{self.update_type}_test.json"
            self.query_sql_file = CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.update_type}/{self.dataset_name}.sql")
        elif self.workload_type:
            self.query_json_file_source = CURRENT_DIR / f"data/{self.dataset_name}/workload_updates/query_{self.workload_type}.json"
            self.query_sql_file = CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.workload_type}/{self.dataset_name}.sql")
        else:
            self.query_json_file_source = CURRENT_DIR / f"data/{self.dataset_name}/query.json"
            self.query_sql_file = CURRENT_DIR / Path(f"workloads/{self.dataset_name}/{self.dataset_name}.sql")

        if not self.query_json_file_source.exists():
            raise FileNotFoundError(f"Query json file not found: {self.query_json_file_source}")

        # prediction paths
        if self.model_name.is_ours():
            if self.update_type:
                pattern = str(CURRENT_DIR / f"data/excels/{self.dataset_name}/*{self.dataset_name}_test_sample_*{self.update_type}*")
                self.prediction_paths_source = [Path(p) for p in glob(pattern)]
                no_of_predictions = len(self.prediction_paths_source)
                assert no_of_predictions < 4, f"You cannot have more than 4 predictions for our model got [{no_of_predictions}]s - {self.prediction_paths_source}"
                self.prediction_paths_destination = [Path(f"workloads/{self.dataset_name}_{self.update_type}/estimates/{self.model_name.name.lower()}_{self.get_type_of_retrain(pp)}.csv")  for pp in self.prediction_paths_source]
            elif self.workload_type:
                pattern = str(CURRENT_DIR / f"data/excels/*{self.dataset_name}_test_sample_*{self.workload_type}*")
                self.prediction_paths_source = [Path(p) for p in glob(pattern)]
                self.prediction_paths_destination = [Path(f"workloads/{self.dataset_name}_{self.workload_type}/estimates/{self.model_name.name.lower()}.csv")  for pp in self.prediction_paths_source]
            elif self.custom_update_type:
                pattern = str(CURRENT_DIR / f"data/excels/{self.dataset_name}/*{self.dataset_name}_test_sample_*{self.custom_update_type}*")
                self.prediction_paths_source = [Path(p) for p in glob(pattern)]
                self.prediction_paths_destination = [Path(f"workloads/{self.dataset_name}/estimates/{self.model_name.name.lower()}_{pp.stem.split('_')[6]}.csv")  for pp in self.prediction_paths_source]
            else:
                self.prediction_paths_source = [CURRENT_DIR / f"data/excels/{self.dataset_name}/dvine_v1_{self.dataset_name}_test_sample.xlsx"]
                self.prediction_paths_destination = [CURRENT_DIR / Path(f"workloads/{self.dataset_name}/estimates/{self.model_name.name.lower()}.csv")]
        else:
            # self.prediction_path = Prerequisists.EXTERNAL_PRED_PATH /  f"/output/result/{self.dataset_name}/{self.update_type}_existing/updated_model/"
            if self.update_type:
                pattern_1 = str(Prerequisists.EXTERNAL_PRED_PATH / f"output/result/{self.dataset_name}/{self.update_type}_existing_model/*{self.model_name}*.csv")
                pattern_2 = str(Prerequisists.EXTERNAL_PRED_PATH / f"output/result/{self.dataset_name}/{self.update_type}_updated_model/*{self.model_name}*.csv")
                self.prediction_paths_source = [Path(p) for p in glob(pattern_1)] + [Path(p) for p in glob(pattern_2)]
                no_of_predictions = len(self.prediction_paths_source)
                assert no_of_predictions < 3, f"You cannot have more than 2 predictions for ACE model got [{no_of_predictions}]s - {self.prediction_paths_source}"
                self.prediction_paths_destination = [CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.update_type}/estimates/{self.model_name.name.lower()}_existing.csv"), 
                                                     CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.update_type}/estimates/{self.model_name}_updated.csv")]
            elif self.workload_type:
                pattern = str(Prerequisists.EXTERNAL_PRED_PATH / f"output/result/{self.dataset_name}/{self.workload_type}/*{self.model_name}*.csv")
                self.prediction_paths_source = [Path(p) for p in glob(pattern)]
                # no_of_predictions = len(self.prediction_paths_source)
                # assert no_of_predictions < 3, f"You cannot have more than 2 predictions for ACE model got [{no_of_predictions}]s - {self.prediction_paths_source}"
                self.prediction_paths_destination = [CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.workload_type}/estimates/{self.model_name.name.lower()}.csv")]
            else:
                pattern = str(Prerequisists.EXTERNAL_PRED_PATH / f"output/result/{self.dataset_name}_{self.dataset_name}/*{self.model_name.name}*.csv")
                self.prediction_paths_source = [Path(p) for p in glob(pattern)]
                # TODO: Later
                assert True, "This is not implemented"
                self.prediction_paths_destination = [CURRENT_DIR / Path(f"workloads/{self.dataset_name}/estimates/{self.model_name.name.lower()}.csv")]

        # if prediction paths are empty, raise an error
        if not self.prediction_paths_source:
            raise ValueError(f"No predictions found for {self.dataset_name} with {self.model_name} and {self.update_type} and {self.workload_type} and {self.custom_update_type}")
        
        for prediction_path in self.prediction_paths_source:
            if not prediction_path.exists():
                raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
            
        # original dataset paths
        if self.dataset_name.is_tpch_type():
            self.original_dataset_path_source = get_data_path(self.dataset_name.value) / "original.parquet"
            if not self.original_dataset_path_source.exists():
                # Try alternative path if original.csv does not exist
                alt_path = Path(f"{Prerequisists.EXTERNAL_PRED_PATH}/data/{self.dataset_name.value}/original.parquet")
                if alt_path.exists():
                    self.original_dataset_path_source = alt_path
        else:
            self.original_dataset_path_source = self.dataset_name.get_file_path(self.update_type) if self.update_type == None else self.dataset_name.get_file_path(filename=f"data_updates/original_{self.update_type}.parquet")
        if not self.original_dataset_path_source.exists():
            raise FileNotFoundError(f"Original dataset file not found: {self.original_dataset_path_source}")
        if self.update_type:
            self.original_dataset_path_destination = CURRENT_DIR / Path(f"single_table_datasets/{self.dataset_name}/{self.dataset_name}_{self.update_type}.csv")
        else:
            self.original_dataset_path_destination = CURRENT_DIR / Path(f"single_table_datasets/{self.dataset_name}/{self.dataset_name}.csv")

        # True cardinality paths
        if self.update_type:
            self.true_cardinality_path_source = CURRENT_DIR / f"data/{self.dataset_name}/data_updates/label_{self.update_type}_test.csv"
        elif self.workload_type:
            self.true_cardinality_path_source = CURRENT_DIR / f"data/{self.dataset_name}/workload_updates/label_{self.workload_type}_test.csv"
        else:
            self.true_cardinality_path_source = CURRENT_DIR / f"data/{self.dataset_name}/label_test.csv"
        if not self.true_cardinality_path_source.exists():
            raise FileNotFoundError(f"True cardinality file not found: {self.true_cardinality_path_source}")
        
        if self.update_type:
            self.true_cardinality_path_destination = CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.update_type}/estimates/true_card.csv")
        elif self.workload_type:
            self.true_cardinality_path_destination = CURRENT_DIR / Path(f"workloads/{self.dataset_name}_{self.workload_type}/estimates/true_card.csv")
        else:
            self.true_cardinality_path_destination = CURRENT_DIR / Path(f"workloads/{self.dataset_name}/estimates/true_card.csv")
        
        table = Table(title=f"Prerequisists for {self.dataset_name} with {self.model_name} update type: {self.update_type} workload type: {self.workload_type}")
        table.add_column("File Type")
        table.add_column("source/destination")
        table.add_column("File Path")
        table.add_column("Size (MB)")
        table.add_column("Exists")

        table.add_row("Query", "source", str(self.query_json_file_source), f"{get_file_size_in_mb(self.query_json_file_source)}", "True")
        table.add_row("Query", "destination", str(self.query_sql_file), f"{get_file_size_in_mb(self.query_sql_file)}", f"{self.query_sql_file.exists()}")
        table.add_row("Original Dataset", "source", str(self.original_dataset_path_source), f"{get_file_size_in_mb(self.original_dataset_path_source)}", "True")
        table.add_row("Original Dataset", "destination", str(self.original_dataset_path_destination), f"{get_file_size_in_mb(self.original_dataset_path_destination)}", f"{self.original_dataset_path_destination.exists()}")
        table.add_row("True Cardinality", "source", str(self.true_cardinality_path_source), f"{get_file_size_in_mb(self.true_cardinality_path_source)}", "True")
        table.add_row("True Cardinality", "destination", str(self.true_cardinality_path_destination), f"{get_file_size_in_mb(self.true_cardinality_path_destination)}", f"{self.true_cardinality_path_destination.exists()}")

        for index, (prediction_path_source, prediction_path_destination) in enumerate(zip(self.prediction_paths_source, self.prediction_paths_destination)):
            table.add_row("Prediction", f"source-{index + 1}", str(prediction_path_source), f"{get_file_size_in_mb(prediction_path_source)}", "True")
            table.add_row("Prediction", f"destination-{index + 1}", str(prediction_path_destination), f"{get_file_size_in_mb(prediction_path_destination)}", f"{prediction_path_destination.exists()}")
        
        self.no_of_rows = self.get_dataset_no_of_rows()
        
        console = Console()
        console.print(table)
        

    def create_all_directories(self):
        all_destination_directories = [self.query_sql_file.parent, self.original_dataset_path_destination.parent, self.true_cardinality_path_destination.parent] + [f.parent for f in self.prediction_paths_destination]
        for destination_directory in all_destination_directories:
            destination_directory.mkdir(parents=True, exist_ok=True)

    def get_dataset_no_of_rows(self):
        df = load_dataframe(self.original_dataset_path_source)
        return df.shape[0]

    def get_type_of_retrain(self, prediction_path):
        if "retrained" in prediction_path.name:
            type_of_rt = TypeOfRetrain.RETRAINED
        elif "updated" in prediction_path.name:
            type_of_rt = TypeOfRetrain.UPDATED
        else:
            type_of_rt = TypeOfRetrain.NONE
        
        if type_of_rt not in self.list_of_retrained_models:
            self.list_of_retrained_models.append(type_of_rt)
        else:
            raise ValueError(f"You cannot have same type of retrain - {self.prediction_paths_source}")
        return type_of_rt
        

    def create_queries(self):
        """
        Create the queries for the dataset.
        """
        ret = json_file_to_sql_file(self.query_json_file_source, self.query_sql_file, self.dataset_name)
        assert ret, "Failed to create queries"


    def copy_predictions(self):
        """
        Copy the predictions for the dataset.
        """
        for prediction_path_source, prediction_path_destination in zip(self.prediction_paths_source, self.prediction_paths_destination):
            if self.model_name.is_ours():
                ret = excel_to_estimates_csv(prediction_path_source, prediction_path_destination, no_of_rows=self.no_of_rows)
            else:
                ret = csv_to_estimates_csv(prediction_path_source, prediction_path_destination, no_of_rows=self.no_of_rows)
            
            remove_header_if_exists(prediction_path_destination.as_posix())
            assert ret, "Failed to copy predictions"
    

    def copy_original_dataset(self):
        """
        Copy the original dataset for the dataset.
        """
        if self.original_dataset_path_source.suffix == ".csv":
            shutil.copy(self.original_dataset_path_source, self.original_dataset_path_destination)
        else:
            df = load_dataframe(self.original_dataset_path_source)
            save_dataframe(df, self.original_dataset_path_destination)
    
    def copy_true_cardinality(self):
        """
        Copy the true cardinality for the dataset.
        """
        shutil.copy(self.true_cardinality_path_source, self.true_cardinality_path_destination)
        remove_header_if_exists(self.true_cardinality_path_destination.as_posix())


    def execute(self, user_input: bool = False):
        """
        Execute the prerequisites for the dataset.
        """
        logger.info(f"Creating queries for {self.dataset_name} with {self.model_name} and {self.update_type}")
        # confirm with user input
        if user_input and input(f"Are you sure you want to create queries for {self.dataset_name} with {self.model_name} and {self.update_type}? (y/n): ") != "y":
            logger.info("Exiting...")
            return
        logger.info("Creating all directories")
        self.create_all_directories()
        logger.info("Creating queries")
        self.create_queries()
        logger.info("Copying predictions")
        self.copy_predictions()
        logger.info("Copying original dataset")
        self.copy_original_dataset()
        logger.info("Copying true cardinality")
        self.copy_true_cardinality()


def main():
    # param_list = []
    # for dataset_name in DatasetNames:
    #     for model_name in ModelNames:
    #         for update_type in UpdateTypes:
    #             param = (dataset_name, model_name, update_type)
    #             param_list.append(param)
    # df = pd.DataFrame(param_list)
    # df.to_csv("param_list.csv", index=False)

    # load last index from file
    try:
        with open("last_index.txt", "r", encoding="utf-8") as f:
            last_index = int(f.read().strip())
    except FileNotFoundError:
        last_index = 0
    logger.info(f"Last index: {last_index}")


    df = pd.read_csv("param_list.csv", header=None)
    # for index, row in df.iterrows():
    for index, row in df.iloc[last_index:].iterrows():
        # try:  
        dataset_name, model_name, update_type = row
        prerequisists = Prerequisists(dataset_name, update_type, model_name)
        prerequisists.execute(user_input=True)
        # except Exception as e:
        #     logger.error(f"Error for {dataset_name} with {update_type} and {model_name}: {e}")
        
        # save last index to a file
        with open("last_index.txt", "w", encoding="utf-8") as f:
            f.write(str(index))
        logger.info(f"Completed for {dataset_name} with {update_type} and {model_name}")
        logger.info(f"Last index saved: {index}")
    logger.info("All done!")





if __name__ == "__main__":
    args = docopt(__doc__)
    
    _dataset_name = args['<dataset_name>']
    _model_name = args['<model_name>']
    _update_type = args['<update_type>']
    
    # Create prerequisites instance and execute
    prerequisists = Prerequisists(_dataset_name, _model_name, _update_type)
    prerequisists.execute(user_input=False)

    # main()