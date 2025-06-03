import numpy as np


def qerror(est_card, card, no_of_rows=None):
    if no_of_rows is not None:
        est_card = est_card * no_of_rows
        est_card = max(est_card, 1)
        card = card * no_of_rows
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


def qerror_batch(est_card, card, no_of_rows=None):
    return [qerror(est, c, no_of_rows) for est, c in zip(est_card, card)]


if __name__ == "__main__":
    print(qerror(0, 0))
