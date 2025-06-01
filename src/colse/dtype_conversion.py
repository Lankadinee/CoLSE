


import numpy as np


def convert_to_low_precision_dtype(array: np.ndarray) -> np.ndarray:
    """
    Convert a given dtype to a lower precision dtype.
    """
    max_val = np.max(array)
    is_int = np.equal(np.mod(array, 1), 0).all()
    is_unsigned = np.min(array) >= 0

    if is_int:
        if is_unsigned:
            if max_val <= np.iinfo(np.uint8).max:
                return array.astype(np.uint8)
            elif max_val <= np.iinfo(np.uint16).max:
                return array.astype(np.uint16)
            elif max_val <= np.iinfo(np.uint32).max:
                return array.astype(np.uint32)
            elif max_val <= np.iinfo(np.uint64).max:
                return array.astype(np.uint64)
        else:
            if np.iinfo(np.int8).min <= np.min(array) and max_val <= np.iinfo(np.int8).max:
                return array.astype(np.int8)
            elif np.iinfo(np.int16).min <= np.min(array) and max_val <= np.iinfo(np.int16).max:
                return array.astype(np.int16)
            elif np.iinfo(np.int32).min <= np.min(array) and max_val <= np.iinfo(np.int32).max:
                return array.astype(np.int32)
            elif np.iinfo(np.int64).min <= np.min(array) and max_val <= np.iinfo(np.int64).max:
                return array.astype(np.int64)
    else:
        if max_val <= np.finfo(np.float16).max:
            return array.astype(np.float16)
        elif max_val <= np.finfo(np.float32).max:
            return array.astype(np.float32)
        elif max_val <= np.finfo(np.float64).max:
            return array.astype(np.float64)