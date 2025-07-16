import json
import numpy as np
import torch
from loguru import logger


from colse.res_utils import decode_label, encode_label, multiply_pairs_norm
from colse.residual_data_conversion import ResidualData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    logger.info(f"Number of model parameters: {num_params} (~= {mb:.2f}MB)")
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


def evaluate(preds, labels, total_rows=-1, verbose=False):
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
    }

    if total_rows > 0:
        metrics["rms"] = rmserror(preds, labels, total_rows) 
    logger.info(f"{json.dumps(metrics)}") if verbose else None
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

    # This is the old logic
    # valid_preds = np.maximum(
    #     np.round(decode_label(pred_np[:, 2])), 0.0
    # ) * valid_preds_sign + np.maximum(np.round(decode_label(y_bar_np)), 0.0)
    # return valid_preds

    # This is the new logic
    pred_label = np.maximum(np.round(decode_label(pred_np[:, 2])), 0.0)
    y_bar_label = np.maximum(np.round(decode_label(y_bar_np)), 0.0)
    valid_preds = pred_label * valid_preds_sign + y_bar_label
    # valid_preds = np.round(decode_label(pred_np[:, 2])) * valid_preds_sign + np.round(decode_label(y_bar_np))
    return np.maximum(valid_preds, 0.0)


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
