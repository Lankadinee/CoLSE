from sklearn.preprocessing import MinMaxScaler

from cest.datasets.dataset import Dataset


def get_mm_scaled_dataset(dataset: Dataset) -> (Dataset, MinMaxScaler):
    scaler = MinMaxScaler()
    scaler.fit(dataset.data[dataset.continuous_columns])
    dataset.data[dataset.continuous_columns] = scaler.transform(dataset.data[dataset.continuous_columns])
    return dataset, scaler


def test():
    from cest.datasets.employee_dataset import load_employee_data

    dataset = load_employee_data()
    print(dataset.data.head())
    dataset, scaler = get_mm_scaled_dataset(dataset)
    print(dataset.data.head())
    print(scaler)


if __name__ == '__main__':
    test()
