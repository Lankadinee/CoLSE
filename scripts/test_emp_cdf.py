

import pandas as pd
from loguru import logger
import numpy as np
from colse.cat_transform import DeQuantize
from colse.cdf_dataframe import CDFDataFrame
from colse.data_path import get_data_path
from colse.df_utils import load_dataframe, save_dataframe
from colse.emphirical_cdf import EMPMethod, EmpiricalCDFModel
from colse.optimized_emp_cdf import OptimizedEmpiricalCDFModel
from colse.spline_dequantizer import SplineDequantizer
from colse.transform_datasets import convert_df_to_dequantize

col_name = "Record_Type"
values = ["BOAT", "SNOW", "TRL "]

def main():
    df1 = load_dataframe(get_data_path() / "dmv/dmv.csv")
    df2 = load_dataframe("/home/dulanj/Learn/dinee/megadrive/query-optimization-methods/library/data/dmv/dmv_dequantized_10KU.parquet")
    col_id = df1.columns.get_loc(col_name)
    dequantize = DeQuantize()
    mapping = dequantize.fit(df1.iloc[:, col_id].to_numpy()).mapping
    cdf_df = CDFDataFrame(OptimizedEmpiricalCDFModel, max_unique_values="auto")
    cdf_df.fit(df2, nproc=1)

    lower_bounds = []
    upper_bounds = []
    for v in values:
        nv = mapping[v]
        logger.info(f"Original value: {v}, New value: {nv}")
        lower_bounds.append(nv[0])
        upper_bounds.append(nv[1])
    
    lb_cdf = cdf_df.predict(lower_bounds, col_name)
    ub_cdf = cdf_df.predict(upper_bounds, col_name)
    print(lb_cdf)
    print(ub_cdf)

    cdf_diff = ub_cdf - lb_cdf
    print(cdf_diff)



def main2():
    df1 = load_dataframe(get_data_path() / "dmv/dmv.csv")
    df2 = load_dataframe(get_data_path() / "dmv/dmv_dequantized_v2.parquet")

    dequantizer = SplineDequantizer(M=10000)
    dequantizer.fit(df1, cat_cols=[col_name])

    cdf_df = CDFDataFrame(OptimizedEmpiricalCDFModel, max_unique_values="auto")
    cdf_df.fit(df2, nproc=1)

    lower_bounds = []
    upper_bounds = []
    for v in values:
        nv = dequantizer.get_interval(col_name, v)
        logger.info(f"Original value: {v}, New value: {nv}")
        lower_bounds.append(nv[0])
        upper_bounds.append(nv[1])
    
    lb_cdf = cdf_df.predict(lower_bounds, col_name)
    ub_cdf = cdf_df.predict(upper_bounds, col_name)
    print(lb_cdf)
    print(ub_cdf)

    cdf_diff = ub_cdf - lb_cdf
    print(cdf_diff)



def create_custom_test_data():
    import pandas as pd
    
    # Create a two column dataframe
    df = pd.DataFrame({
        'col1': range(1, 1001),  # 1 to 1000
        'col2': range(1000, 0, -1)  # 1000 to 1
    })
    
    save_dataframe(df, get_data_path() / "dmv/custom_test.csv")

def create_categorical_test_data():
    import pandas as pd
    
    # Create a two column dataframe with categorical data
    categories_col1 = ['A', 'B', 'C', 'D', 'E']
    categories_col2 = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange']
    
    # Create 1000 rows with random categorical values
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'cat_col1': np.random.choice(categories_col1, size=1000),
        'cat_col2': np.random.choice(categories_col2, size=1000)
    })
    
    save_dataframe(df, get_data_path() / "dmv/categorical_test.csv")



def main3():
    # values = [1, 2, 5, 10, 50, 100, 500, 1000]
    # col_name = "col1"
    # df = load_dataframe(get_data_path() / "dmv/custom_test.csv")

    values = ['A', 'B', 'C', 'D', 'E']
    col_name = "cat_col1"
    df = load_dataframe(get_data_path() / "dmv/categorical_test.csv")

    dequantizer = SplineDequantizer(M=10000)
    dequantizer.fit(df, cat_cols=[col_name])

    df2 = dequantizer.transform(df)

    cdf_df = CDFDataFrame(OptimizedEmpiricalCDFModel, max_unique_values="auto")
    cdf_df.fit(df2, nproc=1)

    lower_bounds = []
    upper_bounds = []
    for v in values:
        nv = dequantizer.get_interval(col_name, v)
        logger.info(f"Original value: {v}, New value: {nv}")
        lower_bounds.append(nv[0])
        upper_bounds.append(nv[1])
    
    lb_cdf = cdf_df.predict(lower_bounds, col_name)
    ub_cdf = cdf_df.predict(upper_bounds, col_name)
    print(lb_cdf)
    print(ub_cdf)

    cdf_diff = ub_cdf - lb_cdf
    print(cdf_diff)


def main4():
    values = [1, 2, 5, 10, 50, 100, 500, 1000]
    col_name = "col1"
    df = load_dataframe(get_data_path() / "dmv/custom_test.csv")

    # values = ['A', 'B', 'C', 'D', 'E']
    # col_name = "cat_col1"
    # df = load_dataframe(get_data_path() / "dmv/categorical_test.csv")

    dequantize = DeQuantize()
    dequantize.fit(df.iloc[:, 0].to_numpy())

    df2 = convert_df_to_dequantize(df, parellel=False, col_list_to_be_dequantized=None, col_list_to_exclude=None)
    mapping = dequantize.fit(df.iloc[:, 0].to_numpy()).mapping
    cdf_df = CDFDataFrame(OptimizedEmpiricalCDFModel, max_unique_values="auto")
    cdf_df.fit(df2, nproc=1)

    lower_bounds = []
    upper_bounds = []
    for v in values:
        nv = mapping[v]
        logger.info(f"Original value: {v}, New value: {nv}")
        lower_bounds.append(nv[0])
        upper_bounds.append(nv[1])
    
    lb_cdf = cdf_df.predict(lower_bounds, col_name)
    ub_cdf = cdf_df.predict(upper_bounds, col_name)
    print(lb_cdf)
    print(ub_cdf)

    cdf_diff = ub_cdf - lb_cdf
    print(cdf_diff)



if __name__ == "__main__":
    # main()
    # main2()
    # create_custom_test_data()
    # create_categorical_test_data()
    main4()
