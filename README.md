# CoLSE - Copula based Learned Selectivity Estimator


CoLSE is a machine learning-based cardinality estimation system that uses copula models to capture dependencies between database columns for accurate selectivity estimation.

## Installation Guide

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd CoLSE
   ```
2. **Install Just**
   ```bash
   brew install just
   ```

## Usage

### Download Required Data

Run the following command to download essential datasets and resources:
```bash
just download
```

Please use the following link Tto download all the datasets used to condcut the experiments: [data-drive](https://mega.nz/folder/dGEF3KaR#9uGAu2EhfKHZJphr1BANUA)

### Install Dependencies

Install all required Python dependencies and set up the development environment.

```bash
just install
```

### Train the Model

Train the model using:
```bash
just train
```
This will start the training process for the selectivity estimator.

### Test the Model

Evaluate the trained model with:
```bash
just test
```
This will run the test suite to validate model performance.

### Run All Steps (Download, Train, Test)

To execute the full pipeline (download, train, and test):
```bash
just all
```
This will sequentially download data, train the model, and test it.

## Check accuracy using no of optimnal query plans

### Prepare the data for the run

```
just prepare-data <dataset_name> <model_name> <update_type>
```

ex: just prepare-data dmv dvine ind_0.2


### Build docker images

```
just build-postgres
```

