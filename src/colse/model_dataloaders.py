# convert parameter dict of lw(nn)


from loguru import logger
from torch.utils.data import Dataset

from colse.model_utils import convert_to_residual
from colse.residual_data_conversion import DataConversion


class LWQueryDataset(Dataset):
    def __init__(self, X, y, gt):
        super(LWQueryDataset, self).__init__()
        self.X = X
        self.y = y
        self.gt = gt

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.gt[idx]


def make_dataset(dataset, num=-1):
    X, y, gt, name = dataset
    logger.info(f"{X.shape}, {y.shape}, {gt.shape}")
    if num <= 0:
        return LWQueryDataset(X, y, gt)
    else:
        logger.info(f"Trimming {name} dataset to {num}")
        return LWQueryDataset(X[:num], y[:num], gt[:num])


def load_lw_dataset(args, excel_path_train, excel_path_valid=None):
    dc = DataConversion(dataset_name=args.dataset)
    rd = dc.convert(excel_path_train, use_cache=False)
    x, y, gt = convert_to_residual(rd)

    logger.info("Data preparation complete")
    train_size = int(args.train_test_split * x.shape[0])

    dataset = {}
    dataset["train"] = (x[:train_size], y[:train_size], gt[:train_size], "train")
    if excel_path_valid is None:
        dataset["valid"] = (x[train_size:], y[train_size:], gt[train_size:], "valid")
    else:
        logger.info("Loading validation dataset")
        rd_valid = dc.convert(excel_path_valid, use_cache=False)
        x_valid, y_valid, gt_valid = convert_to_residual(rd_valid)
        dataset["valid"] = (x_valid, y_valid, gt_valid, "valid")
    return dataset
