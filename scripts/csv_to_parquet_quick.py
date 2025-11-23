import pandas as pd
df = pd.read_csv("data/auction/bid_history_all_2025-11-21.csv")
df.to_parquet("data/auction/bid_history_all_2025-11-21.parquet", engine="pyarrow", compression="snappy")