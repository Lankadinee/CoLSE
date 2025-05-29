from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from colse.datasets.params import ROW_PREFIX, SAMPLE_PREFIX


current_dir = Path(__file__).parent
dataset_dir = current_dir.joinpath("../../data/gas_data")

def generate_dataset(**kwargs):
    nrows = kwargs.get('no_of_rows', 500_000)
    """ Load dataset"""

    df = pd.read_hdf(dataset_dir.joinpath("gas_discrete.hdf"), key="dataframe")

    nrows = df.shape[0] if nrows is None else nrows

    attr1 = df['Time'].to_numpy().astype(int)[:nrows]
    attr2 = df['Humidity'].to_numpy().astype(int)[:nrows]
    attr3 = df['Temperature'].to_numpy().astype(int)[:nrows]
    attr4 = df['Flow_rate'].to_numpy().astype(int)[:nrows]
    attr5 = df['Heater_voltage'].to_numpy().astype(int)[:nrows]
    attr6 = df['R1'].to_numpy().astype(int)[:nrows]
    attr7 = df['R5'].to_numpy().astype(int)[:nrows]
    attr8 = df['R7'].to_numpy().astype(int)[:nrows]
    # data = df.to_numpy().astype(int)

    # Stack the attributes into a 2D array
    data = np.column_stack(
        (attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8))

    df = pd.DataFrame(data, columns=[f"{ROW_PREFIX}{i}" for i in range(1, 9)])
    # print(df.shape)
    """convert all the data into float64"""
    df = df.astype(np.float64)
    return df


def get_queries(**kwargs):
    query_type = kwargs.get('query_type', 'train')
    no_of_queries = kwargs.get('no_of_queries', None)
    data_split = kwargs.get('data_split', None)


    query_l = np.load(dataset_dir.joinpath("query_left_sc.npy").as_posix())
    query_r = np.load(dataset_dir.joinpath("query_right_sc.npy").as_posix())
    true_card = np.load(dataset_dir.joinpath("query_true_sc.npy").as_posix())

    if no_of_queries is not None:
        query_l = query_l[:no_of_queries]
        query_r = query_r[:no_of_queries]
        true_card = true_card[:no_of_queries]
    else:
        """convert all the data into float64"""
        query_l = query_l.astype(np.float64)
        query_r = query_r.astype(np.float64)
        true_card = true_card.astype(np.float64)

    if data_split:
        index_list = np.arange(len(query_l))
        training_indices, test_indices = train_test_split(index_list, test_size=0.2, random_state=42)
        if data_split == 'test':
            return query_l[test_indices], query_r[test_indices], true_card[test_indices]
        else:
            return query_l[training_indices], query_r[training_indices], true_card[training_indices]
    else:
        return query_l, query_r, true_card


if __name__ == '__main__':
    _df, dfs = generate_dataset()
    print(_df.head())
    print(_df.shape)
    print(dfs.head())
    print(dfs.shape)
