import pandas as pd
from colse.df_utils import load_dataframe
from pathlib import Path

model_name = "mhist"
dataset_name = "tpch_sf2_z1_lineitem-old"
result_file_path = f"data/excels/original-base-mhist-bins=30000.csv"
df = load_dataframe(result_file_path)

estimates = df["predict"].to_numpy()

output_dir = Path("output") / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)
output_file_path = output_dir / f"{model_name}.txt"
with open(output_file_path, "w") as f:
    for est in estimates:
        f.write(str(int(est)))
        f.write("\n")
print("Saved to ", output_file_path)

