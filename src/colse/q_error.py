import numpy as np
import pandas as pd

from colse.df_utils import load_dataframe


def qerror(*, est_card, card, no_of_rows=None):
    if no_of_rows is not None:
        est_card = est_card * no_of_rows
        est_card = max(est_card, 1)
        card = card * no_of_rows
    if card > 1:
        pass
    else:
        est_card = np.clip(est_card, 0, 1)

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


def qerror_batch_old(est_card, card, no_of_rows=None):
    return [qerror(est_card=est, card=c, no_of_rows=no_of_rows) for est, c in zip(est_card, card)]

def qerror_np(est_card, card, no_of_rows=None):
    """
    Vectorized Q-error. Accepts scalars or array-like; returns a NumPy array.
    Semantics match `qerror` element-wise.
    """
    est = np.asarray(est_card, dtype=float)
    tru = np.asarray(card, dtype=float)

    if no_of_rows is not None:
        est = est * no_of_rows
        est = np.maximum(est, 1.0)
        tru = tru * no_of_rows

    # If tru <= 1, clip est to [0, 1]; otherwise leave as-is (element-wise)
    est = np.where(tru > 1.0, est, np.clip(est, 0.0, 1.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.divide(est, tru, out=np.full_like(est, np.inf), where=tru != 0)
        r2 = np.divide(tru, est, out=np.full_like(est, np.inf), where=est != 0)
        q = np.maximum(r1, r2)

    both_zero = (est == 0) & (tru == 0)
    est_zero  = (est == 0) & (tru != 0)
    tru_zero  = (tru == 0) & (est != 0)

    q = q.astype(float, copy=False)
    q[both_zero] = 1.0
    q[est_zero]  = tru[est_zero]
    q[tru_zero]  = est[tru_zero]
    return q


def qerror_batch(est_card, card, no_of_rows=None):
    """
    Batch Q-error that now accepts NumPy arrays or lists.
    Returns a Python list for backward-compatibility.
    """
    return qerror_np(est_card=est_card, card=card, no_of_rows=no_of_rows).tolist()


if __name__ == "__main__":
    dataset_name = "dmv"
    true_card = pd.read_csv(f"workloads/{dataset_name}/estimates/true_card.csv", header=None)
    pred_card = pd.read_csv(f"workloads/{dataset_name}/estimates/colse_no_error_comp.csv", header=None)
    true_card = true_card.iloc[:, 1].values
    pred_card = pred_card.iloc[:, 1].values
    new_q_errors = qerror_batch(est_card=pred_card, card=true_card, no_of_rows=11591877)
    old_q_errors = qerror_batch_old(est_card=pred_card, card=true_card, no_of_rows=11591877)

    """Find the index where the new and old q_errors are the different, remove small differences"""
    different_indices = np.where(np.array(new_q_errors) != np.array(old_q_errors))[0]
    different_indices = different_indices[np.abs(np.array(new_q_errors)[different_indices] - np.array(old_q_errors)[different_indices]) > 1e-6]
    print(len(different_indices))
    print(different_indices)
    pass