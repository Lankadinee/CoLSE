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
from torch.utils.data import DataLoader

from colse.data_path import get_log_path, get_model_path
from colse.error_comp_model import ErrorCompModel
from colse.model_dataloaders import load_lw_dataset, make_dataset
from colse.model_utils import (
    batch_qerror,
    calculate_class_weights,
    evaluate,
    get_actual_cardinality,
    report_model,
)
from colse.res_utils import decode_label
from default_args import Args

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count()))
current_dir = Path(__file__).resolve().parent
iso_time_str = pd.Timestamp.now().isoformat()
LOG_ROOT = get_log_path()
logger.add(
    LOG_ROOT.joinpath(f"training-{iso_time_str}.log"),
    rotation="10 MB",
    level="DEBUG",
)


logger = logger
args = None


iso_time_str = pd.Timestamp.now().isoformat()
iso_time_str = iso_time_str.replace(":", "-")


def train_lw_nn(output_model_path, pretrained_model_path, seed=42):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    logger.info(f"torch threads: {torch.get_num_threads()}")

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
    logger.info(f"Overall LWNN model size = {model_size:.2f}MB")

    if pretrained_model_path:
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
        state = torch.load(pretrained_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded pretrained model from {pretrained_model_path}")

    # load dataset
    dataset = load_lw_dataset(
        args=args, excel_path_train=args.train_excel_path, excel_path_valid=None
    )
    train_dataset = make_dataset(
        dataset["train"], num=int(args.no_of_queries * args.train_test_split)
    )
    valid_dataset = make_dataset(
        dataset["valid"], num=int(args.no_of_queries * (1 - args.train_test_split))
    )

    class_weights = calculate_class_weights(train_dataset.y)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    logger.info("Train loader created")
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)
    logger.info("Valid loader created")

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
        logger.info(
            f"Epoch {epoch+1}, loss: {train_loss.mean()}, {trained_loop}/{total_loop}|{trained_loop*100/total_loop:.2f} time since start: {dur_min:.1f} mins"
        )

        # run.log({"epoch": epoch + 1, "train_loss": train_loss.mean()})

        logger.info(f"Test on valid set...")
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
        logger.info(f"Valid loss is {valid_loss:.4f}")
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

        logger.info("Q-Error on validation set:")
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
            logger.info("best valid loss for now!")
            best_valid_loss = valid_loss
            torch.save(get_state(), output_model_path)

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

    logger.info(
        f"Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins"
    )
    logger.info(
        f"Model saved to {output_model_path}, best valid: {state['valid_error']}"
    )


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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hid_units", type=str, default="256_256_128_64", help="Hidden units"
    )
    # train test split
    parser.add_argument(
        "--train_test_split", type=float, default=0.8, help="Train test split ratio"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    # parser add model name
    parser.add_argument(
        "--output_model_name",
        type=str,
        default="error_comp_model.pt",
        help="Model name",
    )
    parser.add_argument(
        "--pretrained_model_name", type=str, default=None, help="Model name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()

    # Convert to dictionary
    args_dict = vars(parsed_args)
    args = Args(**args_dict)

    print(args)
    _output_model_path = get_model_path(args.dataset) / f"{args.output_model_name}"
    _pretrained_model_path = None
    if args.pretrained_model_name:
        _model_path = get_model_path(args.dataset) / f"{args.pretrained_model_name}"
        if _model_path.exists():
            _pretrained_model_path = _model_path
        else:
            logger.error(f"Pretrained model {_model_path} does not exist")
            exit(1)

    train_lw_nn(
        output_model_path=_output_model_path,
        pretrained_model_path=_pretrained_model_path,
    )
