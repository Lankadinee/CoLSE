from loguru import logger
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def compute_mutual_info(X, y, discrete_features='auto'):
    """Compute MI between y and each column in X."""
    if y.dtype.kind in 'if':  # numeric target
        return mutual_info_regression(X, y, discrete_features=discrete_features)
    else:  # categorical target
        return mutual_info_classif(X, y, discrete_features=discrete_features)

def entropy(col):
    """Shannon entropy of a discrete column."""
    probs = col.value_counts(normalize=True)
    return -(probs * np.log2(probs)).sum()

def encode_dataframe(df):
    """Encode categorical columns with LabelEncoder."""
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def mutinfo_ordering(df, method="MutInfo"):
    """
    Column ordering based on mutual information heuristics.
    
    Parameters:
    - df: pandas DataFrame (all categorical/numeric features)
    - method: "MutInfo" or "PMutInfo"
    
    Returns:
    - list of ordered column names
    """
    # get 10% of the dataframe
    df = df.sample(frac=0.1)
    # df.to_excel("mtinfo_input.xlsx")
    df_encoded = encode_dataframe(df)
    cols = df_encoded.columns.tolist()
    chosen = []
    
    # 1. Start with column of maximum entropy
    first_col = max(cols, key=lambda c: entropy(df_encoded[c]))
    chosen.append(first_col)
    remaining = [c for c in cols if c != first_col]
    
    while remaining:
        if method == "MutInfo":
            # Compute MI with joint chosen set
            X = df_encoded[chosen]
            mi_scores = []
            for col in remaining:
                print(f"Mutinfo: {col}")
                score = compute_mutual_info(X, df_encoded[col])[0]  # MI(chosen; candidate)
                mi_scores.append((col, score))
                
        elif method == "PMutInfo":
            # Compute MI with last chosen column
            last = chosen[-1]
            mi_scores = []
            for col in remaining:
                score = compute_mutual_info(df_encoded[[last]], df_encoded[col])[0]
                mi_scores.append((col, score))
        
        # Pick column with highest MI
        best_col = max(mi_scores, key=lambda x: x[1])[0]
        chosen.append(best_col)
        remaining.remove(best_col)
        logger.info(f"{method}: Remaining columns: {remaining}")
    
    return chosen


if __name__ == "__main__":
    # Example DataFrame
    # df = pd.DataFrame({
    #     "city": ["A", "B", "A", "C", "B", "A"],
    #     "year": [2020, 2021, 2020, 2022, 2021, 2020],
    #     "sales": [100, 200, 150, 300, 250, 120]
    # })

    df = pd.read_excel("mtinfo_input.xlsx")

    print(mutinfo_ordering(df, method="MutInfo"))
