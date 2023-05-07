import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
print(df)
print(df.info())
print(df.describe())