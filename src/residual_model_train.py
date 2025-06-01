import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from colse.data_path import get_data_path, get_excel_path, get_log_path, get_model_path
from colse.error_comp_model import ErrorCompModel
from colse.res_utils import decode_label, encode_label, multiply_pairs_norm
from colse.residual_data_conversion import DataConversion, ResidualData
from default_args import Args

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count()))
# logger.remove()
current_dir = Path(__file__).resolve().parent
iso_time_str = pd.Timestamp.now().isoformat()
LOG_ROOT = get_log_path()
logger.add(
    LOG_ROOT.joinpath(f"training-{iso_time_str}.log"),
    rotation="1 MB",
    level="DEBUG",
)

# no_of_rows = 581012


L = logger
"""
AVI feature added
"""


# convert parameter dict of lw(nn)


def parse_args():
    temp_args = Args()
    parser = argparse.ArgumentParser(description="Train LWNN model with residuals")
    parser.add_argument(
        "--train_excel_path",
        type=str,
        default=temp_args.train_excel_path,
        help="Path to the training Excel file",
    )
    parser.add_argument(
        "--test_excel_path",
        type=str,
        default=temp_args.test_excel_path,
        help="Path to the testing Excel file",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="forest", help="Name of the dataset"
    )
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hid_units", type=str, default="256_256_128_64", help="Hidden units"
    )
    # train test split
    parser.add_argument(
        "--train_test_split", type=float, default=0.8, help="Train test split ratio"
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of epochs"
    )
    return parser.parse_args()


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
    L.info(f"{X.shape}, {y.shape}, {gt.shape}")
    if num <= 0:
        return LWQueryDataset(X, y, gt)
    else:
        logger.info(f"Trimming {name} dataset to {num}")
        return LWQueryDataset(X[:num], y[:num], gt[:num])


def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    L.info(f"Number of model parameters: {num_params} (~= {mb:.2f}MB)")
    # L.info(model)
    return mb


def qerror(est_card, card):
    if est_card == 0 and card == 0:
        return 1.0
    if est_card == 0:
        return card
    if card == 0:
        return est_card
    if est_card > card:
        return est_card / card
    else:
        return card / est_card


def batch_qerror(est_cards, cards):
    return np.array([qerror(est, card) for est, card in zip(est_cards, cards)])


def rmserror(preds, labels, total_rows):
    return np.sqrt(np.mean(np.square(preds / total_rows - labels / total_rows)))


def evaluate(preds, labels, total_rows=-1):
    errors = []
    for i in range(len(preds)):
        errors.append(qerror(float(preds[i]), float(labels[i])))

    metrics = {
        "max": np.max(errors),
        "99th": np.percentile(errors, 99),
        "95th": np.percentile(errors, 95),
        "90th": np.percentile(errors, 90),
        "median": np.median(errors),
        "mean": np.mean(errors),
        # "gmean": gmean(errors),
    }

    if total_rows > 0:
        metrics["rms"] = rmserror(preds, labels, total_rows)
    L.info(f"{metrics}")
    return np.array(errors), metrics


def is_good_model(matrix):
    pcnt_max = 3000
    pcnt_99 = 15
    pcnt_95 = 5
    pcnt_90 = 3
    pcnt_median = 1.11

    if matrix["max"] > pcnt_max:
        return False
    if matrix["99th"] > pcnt_99:
        return False
    if matrix["95th"] > pcnt_95:
        return False
    if matrix["90th"] > pcnt_90:
        return False
    if matrix["median"] > pcnt_median:
        return False
    return True


def convert_to_residual(rd: ResidualData):
    no_of_rows = rd.no_of_rows
    x = rd.n_query
    y_bar_log = encode_label(rd.y_bar * no_of_rows)
    x_cdf = rd.x_cdf
    gt = rd.gt

    avi_card = np.array(list(map(multiply_pairs_norm, x_cdf))) * no_of_rows
    avi_card_log = encode_label(np.abs(avi_card))
    # avi_res_log = encode_label(np.abs(rd.y_bar*no_of_rows - avi_card))
    y_res = gt - rd.y_bar * no_of_rows
    y_sign_plus = (y_res >= 0).astype(int)
    y_sign_minus = (y_res < 0).astype(int)
    y_abs = encode_label(np.abs(y_res))
    y = np.concatenate(
        [y_sign_plus[:, None], y_sign_minus[:, None], y_abs[:, None]], axis=1
    )
    x = np.concatenate([x, y_bar_log[:, None], avi_card_log[:, None]], axis=1)
    return x, y, gt


def load_lw_dataset(excel_path_train, excel_path_valid=None):
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


def np_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_actual_cardinality(pred, y_bar):
    y_bar_np = y_bar.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    # valid_preds_sign = np.where(np_sigmoid(pred_np[:, 0]) > 0.5, 1, -1)
    positive_sign = pred_np[:, 0]
    negative_sign = pred_np[:, 1]
    valid_preds_sign = np.zeros_like(positive_sign)
    valid_preds_sign[positive_sign > negative_sign] = 1
    valid_preds_sign[positive_sign < negative_sign] = -1
    valid_preds_sign[(positive_sign > 0) * (negative_sign > 0)] = (
        0  # Zero because we are not applying sigmoid
    )
    valid_preds_sign[(positive_sign < 0) * (negative_sign < 0)] = 0

    valid_preds = np.maximum(
        np.round(decode_label(pred_np[:, 2])), 0.0
    ) * valid_preds_sign + np.maximum(np.round(decode_label(y_bar_np)), 0.0)
    return valid_preds


iso_time_str = pd.Timestamp.now().isoformat()
iso_time_str = iso_time_str.replace(":", "-")


def calculate_class_weights(labels):
    """Calculate class weights for positive and negative sign instances."""
    # Extract positive and negative labels
    positive_count = (labels[:, 0] == 1).sum().item()
    negative_count = (labels[:, 1] == 1).sum().item()
    total_count = positive_count + negative_count

    # Calculate weights (inverse of frequency)
    positive_weight = total_count / (2 * positive_count) if positive_count > 0 else 0.0
    negative_weight = total_count / (2 * negative_count) if negative_count > 0 else 0.0

    return torch.tensor([positive_weight, negative_weight], device=DEVICE)


def train_lw_nn(model_file, seed=42):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    model = ErrorCompModel(
        args.fea_num,
        args.hid_units,
        output_len=args.output_len,
        dropout_prob=args.dropout_prob,
    ).to(DEVICE)
    model_size = report_model(model)

    L.info(f"Overall LWNN model size = {model_size:.2f}MB")

    # load dataset
    dataset = load_lw_dataset(
        excel_path_train=args.train_excel_path, excel_path_valid=None
    )
    train_dataset = make_dataset(
        dataset["train"], num=int(args.no_of_queries * args.train_test_split)
    )
    valid_dataset = make_dataset(
        dataset["valid"], num=int(args.no_of_queries * (1 - args.train_test_split))
    )

    class_weights = calculate_class_weights(train_dataset.y)

    L.info(f"Number of training samples: {len(train_dataset)}")
    L.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    L.info("Train loader created")
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)
    L.info("Valid loader created")

    # Train model
    state = {
        "seed": seed,
        "args": args.get_hyperparameters(),
        "device": DEVICE,
        "threads": torch.get_num_threads(),
        "dataset": str(args.dataset.value),
        "version": "v1",
        "workload": "workload",
        "model_size": model_size,
        "fea_num": int(args.fea_num),
    }

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.lr), weight_decay=1e-5
    )
    best_valid_loss = float("inf")
    best_50_percentile = float("inf")

    """write a custom loss function to handle the residual values"""

    def custom_loss(preds, labels, class_weights=None):
        # Extract targets
        sign_targets = labels[:, :2]
        abs_val_targets = labels[:, 2]

        # Ensure absolute value targets are non-negative
        assert torch.all(
            labels[:, 2] >= 0
        ), "Absolute value targets contain negative values"

        # Extract predictions (logits for sign prediction)
        sign_logits = preds[:, :2]  # Use logits directly for BCEWithLogitsLoss
        abs_val_preds = F.relu(preds[:, 2])  # Ensure non-negative values

        # Define separate loss functions
        # class_weights = torch.tensor([1.0, 1.0])
        sign_loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
        abs_loss_fn = nn.MSELoss()

        # Calculate sign loss (BCE)
        sign_loss = sign_loss_fn(sign_logits, sign_targets)

        # Calculate absolute value loss (MSE)
        abs_loss = abs_loss_fn(abs_val_preds, abs_val_targets)

        # Combine the losses
        total_loss = sign_loss + abs_loss

        return total_loss

    start_stmp = time.time()
    valid_time = 0
    tolerance = 1e-3
    for epoch in range(args.epochs):
        train_loss = torch.tensor([])
        model.train()
        total_loop = len(train_loader)
        trained_loop = 0
        for batch_id, data in enumerate(train_loader):
            inputs, labels, gt = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            y_bar_log = inputs[:, -args.additional_features]

            optimizer.zero_grad()
            preds = model(inputs).reshape(-1, args.output_len)

            pred_card = get_actual_cardinality(preds, y_bar_log)
            gt_y_bar = np.maximum(
                np.round(decode_label(y_bar_log.detach().cpu().numpy())), 0.0
            )
            base_error = batch_qerror(gt_y_bar, gt.detach().cpu().numpy())
            combined_error = batch_qerror(pred_card, gt)
            if epoch < 10 or np.mean(combined_error) < np.mean(base_error) - tolerance:
                """Only update the model if the combined error is less than the base error"""
                # logger.info(f"Updating model for epoch {epoch+1}/b{batch_id}")
                loss = custom_loss(preds, labels, class_weights)
                loss.backward()
                optimizer.step()
                trained_loop += 1

                train_loss = torch.cat([train_loss, loss.reshape(-1, 1).cpu()])
        dur_min = (time.time() - start_stmp) / 60
        L.info(
            f"Epoch {epoch+1}, loss: {train_loss.mean()}, {trained_loop}/{total_loop}|{trained_loop*100/total_loop:.2f} time since start: {dur_min:.1f} mins"
        )

        # run.log({"epoch": epoch + 1, "train_loss": train_loss.mean()})

        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        valid_loss = torch.tensor([])
        valid_preds = torch.tensor([])
        valid_y_bar = torch.tensor([])
        valid_gts = torch.tensor([])
        model.eval()
        for _, data in enumerate(valid_loader):
            inputs, labels, gts = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            with torch.no_grad():
                preds = model(inputs).reshape(-1, args.output_len)
                valid_preds = torch.cat([valid_preds, preds.cpu()])
                valid_y_bar = torch.cat(
                    [valid_y_bar, inputs[:, -args.additional_features].cpu()]
                )
                valid_gts = torch.cat([valid_gts, gts.float()])

                # loss = mse_loss(preds, labels)
                loss = custom_loss(preds, labels)
                valid_loss = torch.cat([valid_loss, loss.reshape(-1, 1).cpu()])

        valid_loss = valid_loss.mean()
        L.info(f"Valid loss is {valid_loss:.4f}")
        # During validation, use logits for sign comparison
        positive_sign = valid_preds[:, 0]
        negative_sign = valid_preds[:, 1]

        # Calculate valid_preds_sign based on logits
        valid_preds_sign = np.zeros_like(positive_sign)
        valid_preds_sign[positive_sign > negative_sign] = 1
        valid_preds_sign[positive_sign < negative_sign] = -1
        valid_preds_sign[(positive_sign > 0) & (negative_sign > 0)] = (
            0  # Zero because we are not applying sigmoid
        )
        valid_preds_sign[(positive_sign < 0) & (negative_sign < 0)] = 0

        # valid_preds_sign = np.where(torch.sigmoid(valid_preds[:, 0]) > 0.5, 1, -1)
        valid_pred_abs_np_dec = np.maximum(
            np.round(decode_label(F.relu(valid_preds[:, 2]).detach().cpu().numpy())),
            0.0,
        )
        valid_ybar_np_dec = np.maximum(
            np.round(decode_label(valid_y_bar.detach().cpu().numpy())), 0.0
        )
        valid_preds_f = valid_pred_abs_np_dec * valid_preds_sign + valid_ybar_np_dec

        L.info("Q-Error on validation set:")
        _, metrics = evaluate(valid_preds_f, valid_gts)

        def get_state():
            state["model_state_dict"] = model.state_dict()
            state["optimizer_state_dict"] = optimizer.state_dict()
            state["valid_error"] = {"workload": metrics}
            state["train_time"] = (valid_stmp - start_stmp - valid_time) / 60
            state["current_epoch"] = epoch
            return state

        # """save best model with custom requirements"""
        # if is_good_model(metrics):
        #     L.info("best wify model found!")
        #     L.info("Time to celebrate!")
        #     time.sleep(5)
        #     torch.save(get_state(), model_file)
        #     break

        if valid_loss < best_valid_loss:
            L.info("best valid loss for now!")
            best_valid_loss = valid_loss
            torch.save(get_state(), model_file)

        # """save best 50 percentile matrics"""
        # if metrics["median"] < best_50_percentile:
        #     best_50_percentile = metrics["median"]
        #     torch.save(
        #         get_state(), model_file.parent / f"50_percentile_{model_file.name}"
        #     )
        #     L.info(
        #         f"Best 50 percentile model saved to {model_file.parent / f'50_percentile_{model_file.name}'}"
        #     )


        valid_time += time.time() - valid_stmp

        general_metrics = {
            "valid_loss": valid_loss.item(),
            "epoch": epoch + 1,
            "train_loss": train_loss.mean(),
        }
        general_metrics.update(metrics)

        # run.log(general_metrics)

    L.info(
        f"Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins"
    )
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}")


def get_col_count(x):
    x_group = x.reshape(-1, 2)
    col_count = 0
    for group in x_group:
        if group[0] == 0 and group[1] == 1:
            col_count += 1

    return col_count


def multiply_pairs(x):
    result = 1.0
    for i in range(0, len(x) - 1, 2):
        result *= x[i + 1] - x[i]
    return result * 581012


def evaluate_lw_nn(model_path):
    """load model and evaluate"""
    logger.info(f"Loading model from - {model_path}")
    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = ErrorCompModel(
        state["fea_num"],
        args.hid_units,
        output_len=args.output_len,
        dropout_prob=args.dropout_prob,
    ).to(DEVICE)
    report_model(model)
    L.info(f"Overall LWNN model size = {state['model_size']:.2f}MB")
    model.load_state_dict(state["model_state_dict"])

    df_dict = {}

    dc = DataConversion(dataset_name=args.dataset)
    rd = dc.convert(args.test_excel_path, use_cache=False)
    q_error = rd.q_error
    x, y, gt = convert_to_residual(rd)

    df_dict["query"] = [
        ",".join(list(map(str, list(v)))) for v in x[:, : -args.additional_features]
    ]
    df_dict["gt"] = gt
    df_dict["gt_pred"] = np.maximum(
        np.round(decode_label(x[:, -args.additional_features])), 0.0
    )
    df_dict["q_error"] = q_error

    dataset = LWQueryDataset(x, y, gt)
    valid_loader = DataLoader(dataset, batch_size=1)

    model.eval()
    valid_preds = torch.tensor([])
    valid_y_bar = torch.tensor([])
    col_count = []
    avi_card = []
    all_infer_time = []
    for _, data in enumerate(valid_loader):
        inputs, _, _ = data
        inputs = inputs.to(DEVICE).float()

        with torch.no_grad():
            start_time = time.time()
            preds = model(inputs).reshape(-1, args.output_len)
            infer_time = time.time() - start_time
            all_infer_time.append(infer_time)
            valid_preds = torch.cat([valid_preds, preds.cpu()])
            valid_y_bar = torch.cat(
                [valid_y_bar, inputs[:, -args.additional_features].cpu()]
            )

    L.info(f"Average inference time: {np.mean(all_infer_time):.4f} seconds")

    valid_preds_np = F.relu(valid_preds[:, 2]).detach().cpu().numpy()
    df_dict["v_preds_abs"] = valid_preds_np
    df_dict["v_preds_abs_d"] = np.maximum(np.round(decode_label(valid_preds_np)), 0.0)

    col_count_np = np.array(col_count)
    # Adjust sign prediction logic to use logits
    positive_sign_logits = valid_preds[:, 0]
    negative_sign_logits = valid_preds[:, 1]
    valid_preds_sign = np.zeros_like(positive_sign_logits.cpu())

    valid_preds_sign[positive_sign_logits > negative_sign_logits] = 1
    valid_preds_sign[positive_sign_logits < negative_sign_logits] = -1
    valid_preds_sign[(positive_sign_logits > 0) & (negative_sign_logits > 0)] = 0
    valid_preds_sign[(positive_sign_logits < 0) & (negative_sign_logits < 0)] = 0

    df_dict["v_preds_sign"] = valid_preds_sign

    valid_pred_abs_np_dec = np.maximum(
        np.round(decode_label(F.relu(valid_preds[:, 2]).detach().cpu().numpy())), 0.0
    )
    valid_ybar_np_dec = np.maximum(
        np.round(decode_label(valid_y_bar.detach().cpu().numpy())), 0.0
    )

    valid_preds = valid_pred_abs_np_dec * valid_preds_sign + valid_ybar_np_dec
    df_dict["valid_pred_dec"] = valid_pred_abs_np_dec
    df_dict["valid_preds"] = valid_preds

    L.info("Q-Error on validation set:")
    errors, metrics = evaluate(valid_preds, gt)

    df_dict["errors"] = list(errors)
    df = pd.DataFrame(df_dict)
    excel_file_path = get_excel_path() / f"predictions_v3_{iso_time_str}.xlsx"
    df.to_excel(excel_file_path, index=False)
    logger.info(f"Excel file path : {excel_file_path}")

    """Write valid preds to a txt file"""
    txt_file = (
        get_data_path(args.dataset.value) / f"estimations_sample_auto_max_25000.txt"
    )
    with open(txt_file, "w") as f:
        for vp in valid_preds:
            f.write(str(int(vp)))
            f.write("\n")
    logger.info(f"Saved to {txt_file}")


if __name__ == "__main__":
    parsed_args = parse_args()

    # Convert to dictionary
    args_dict = vars(parsed_args)
    args = Args(**args_dict)

    print(args)
    time.sleep(3)
    model_path = get_model_path(args.dataset)
    model_file = model_path / f"error_comp_model.pt"
    train_lw_nn(model_file)
    # evaluate_lw_nn(model_file)
