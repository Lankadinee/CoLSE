



import numpy as np


def encode_label(label):
    # +1 before log2 to deal with ground truth = 0 scenario
    assert np.all(label >= 0), "All labels should be non-negative"
    return np.log2(label + 1)


def decode_label(label):
    return np.power(2, label) - 1


def multiply_pairs_norm(x):
    result = 1.0
    for i in range(0, len(x) - 1, 2):
        result *= x[i + 1] - x[i]
    return result

if __name__ == "__main__":
    print(decode_label(-0.13))